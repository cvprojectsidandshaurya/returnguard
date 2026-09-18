"""Deployment-ready API for a calibrated ReturnGuard model artifact."""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Response, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
from src.verifier import ReturnVerifier  # noqa: E402

MAX_IMAGE_BYTES = 12 * 1024 * 1024


@dataclass
class Runtime:
    verifier: ReturnVerifier | None
    error: str | None


def load_runtime() -> Runtime:
    checkpoint = os.getenv("RETURNGUARD_CHECKPOINT")
    policy = os.getenv("RETURNGUARD_POLICY")
    if not checkpoint or not policy:
        return Runtime(None, "RETURNGUARD_CHECKPOINT and RETURNGUARD_POLICY must both be configured")
    try:
        return Runtime(ReturnVerifier.from_artifacts(checkpoint, policy, device=os.getenv("RETURNGUARD_DEVICE", "auto")), None)
    except Exception as exc:
        return Runtime(None, f"could not load model artifacts: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = load_runtime()
    yield


app = FastAPI(title="ReturnGuard inference", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health(response: Response) -> dict:
    runtime: Runtime = app.state.runtime
    if runtime.error:
        response.status_code = 503
        return {"ready": False, "error": runtime.error}
    return {"ready": True}


async def save_upload(upload: UploadFile, directory: Path, prefix: str, index: int) -> Path:
    content = await upload.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"{prefix} image {index} exceeds the 12 MB limit")
    suffix = Path(upload.filename or "image.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=415, detail=f"{prefix} image {index} has an unsupported file type")
    path = directory / f"{prefix}_{index}{suffix}"
    path.write_bytes(content)
    return path


@app.post("/verify")
async def verify(
    packing_images: Annotated[list[UploadFile], File(description="seller packing photos")],
    rider_images: Annotated[list[UploadFile], File(description="rider return photos")],
) -> dict:
    runtime: Runtime = app.state.runtime
    if runtime.verifier is None:
        raise HTTPException(status_code=503, detail=runtime.error or "model is unavailable")
    if not packing_images or not rider_images:
        raise HTTPException(status_code=422, detail="at least one packing image and one rider image are required")
    try:
        with tempfile.TemporaryDirectory(prefix="returnguard-") as temp:
            directory = Path(temp)
            packing = [await save_upload(upload, directory, "packing", index) for index, upload in enumerate(packing_images)]
            rider = [await save_upload(upload, directory, "rider", index) for index, upload in enumerate(rider_images)]
            return runtime.verifier.verify(packing, rider).as_dict()
    finally:
        for upload in [*packing_images, *rider_images]:
            await upload.close()
