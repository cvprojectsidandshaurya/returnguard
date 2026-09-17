# ReturnGuard

Computer vision verification for ecommerce returns in India. The seller photographs a garment at packing. The delivery rider photographs it at the doorstep during the existing reverse pickup QC step. Our model compares the two sets and reports MATCH, DIFFERENT_PRODUCT, SUSPICIOUS, or RETAKE.

The research problem is the domain gap: clean flat lay packing shots against crumpled, glare covered, motion blurred rider photos taken on cheap Android phones.

## Layout

```
/ml        notebooks, training, evaluation (Python)
/data      capture protocol, metadata schema, metadata.csv (no images)
/backend   FastAPI inference server
/docs      decisions, results.md, weekly notes
```

`/app` does not exist yet. The framework choice (Expo vs Flutter) is still open, see docs/decisions.md.

## Start here

1. Read `CLAUDE.md` for full project context.
2. Read `data/capture_protocol.md` before photographing anything.
3. Record every experiment as one row in `docs/results.md`.

## Hard rules

- No images, weights, or datasets in git. Photos live in shared storage, git holds `data/metadata.csv` only.
- Splits are by `garment_id` and `design_id`, never by image.
- Every reported number comes from a script with a fixed seed and a recorded commit hash.
- Zero-shot baselines are reported alongside any fine-tuned model.
- Headline metrics are TPR at 1% FPR and Recall@1, with per-condition breakdowns.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt
```
