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
from starlette.concurrency import run_in_threadpool

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
from src.quality import QualityThresholds  # noqa: E402
from src.verifier import ReturnVerifier  # noqa: E402

MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_PACKING_IMAGES = 8
MAX_RIDER_IMAGES = 12


@dataclass
class Runtime:
    verifier: ReturnVerifier | None
    error: str | None


def load_runtime() -> Runtime:
    checkpoint = os.getenv("RETURNGUARD_CHECKPOINT")
    policy = os.getenv("RETURNGUARD_POLICY")
    quality_policy = os.getenv("RETURNGUARD_QUALITY_POLICY")
    if not checkpoint or not policy or not quality_policy:
        return Runtime(None, "RETURNGUARD_CHECKPOINT, RETURNGUARD_POLICY, and RETURNGUARD_QUALITY_POLICY must all be configured")
    try:
        return Runtime(
            ReturnVerifier.from_artifacts(
                checkpoint,
                policy,
                device=os.getenv("RETURNGUARD_DEVICE", "auto"),
                quality_thresholds=QualityThresholds.load(quality_policy),
            ),
            None,
        )
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
    if len(packing_images) > MAX_PACKING_IMAGES:
        raise HTTPException(status_code=422, detail=f"at most {MAX_PACKING_IMAGES} packing images are allowed")
    if len(rider_images) > MAX_RIDER_IMAGES:
        raise HTTPException(status_code=422, detail=f"at most {MAX_RIDER_IMAGES} rider images are allowed")
    try:
        with tempfile.TemporaryDirectory(prefix="returnguard-") as temp:
            directory = Path(temp)
            packing = [await save_upload(upload, directory, "packing", index) for index, upload in enumerate(packing_images)]
            rider = [await save_upload(upload, directory, "rider", index) for index, upload in enumerate(rider_images)]
            result = await run_in_threadpool(runtime.verifier.verify, packing, rider)
            return result.as_dict()
    finally:
        for upload in [*packing_images, *rider_images]:
            await upload.close()
