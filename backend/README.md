# backend

FastAPI inference server for a calibrated ReturnGuard checkpoint.

The API is intentionally configuration-only. It will not start as ready until a
trained checkpoint, a validation-fitted decision policy, and a fitted quality
policy are supplied. Model weights and policies stay out of git.

```bash
pip install -r backend/requirements.txt
export RETURNGUARD_CHECKPOINT=/absolute/path/to/best.pt
export RETURNGUARD_POLICY=/absolute/path/to/policy.json
export RETURNGUARD_QUALITY_POLICY=/absolute/path/to/quality-policy.json
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

`GET /health` reports whether both artifacts loaded. `POST /verify` accepts
multipart lists named `packing_images` and `rider_images` (at most 8 and 12
respectively). A poor rider image returns RETAKE; a poor seller reference returns
REFERENCE_INVALID. Otherwise it returns the calibrated decision, global
multi-view evidence, and optional local-detail evidence. Uploads are held only
in an automatically deleted temporary directory. Put the service behind the
deployment platform's authentication and rate limit before exposing it publicly.

## Container deployment

Build from the repository root after generating `best.pt` and `policy.json`.
The image deliberately contains neither the photos nor model artifacts.

```bash
docker build -f backend/Dockerfile -t returnguard .
docker run --rm -p 8000:8000 \
  --env-file backend/.env.example \
  -v "$PWD/ml/checkpoints:/artifacts:ro" \
  returnguard
```

The mounted artifact directory must contain the exact checkpoint and policy
named in the environment file. `GET /health` should return `{"ready": true}`
before the service is connected to any client or load balancer.
