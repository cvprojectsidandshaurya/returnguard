# Decisions

Append only. Each entry: date, the decision, the alternatives, and why. If a decision is reversed later, add a new entry rather than editing the old one.

## Open

| decision | options | owner | needed by |
| --- | --- | --- | --- |
| App framework | Expo (recommended) vs Flutter | | week 3, before the capture app |
| Rider phones | which budget Androids we buy or borrow | Sid | week 1, blocks capture |
| Image storage | Google Drive vs S3 | Sid | week 1, blocks capture |
| QR tag anchor in V1 | vision only vs vision plus printed tag | Shaurya | week 6 |

## Decided

### 2026-09-16 Monorepo over separate repos

One repo holding ml, data, backend, docs. Two people, tightly coupled work, and the metadata schema is shared between the ML code and the capture tooling. Separate repos would mean version skew between the schema and the code reading it for no gain at this size.

### 2026-09-16 Metadata in git, images outside

`data/metadata.csv` is committed. Images are not, in any form, including thumbnails and sample crops. Git handles thousands of photos badly and the repo becomes unusable once it happens. The cost is that the dataset is only reproducible if the image store is kept in sync, which the metadata's `relative_path` column handles.

### 2026-09-16 Splits by split group, not by garment

Splitting by `garment_id` alone is not enough. Identical units of one design, and lookalikes across brands, must stay on the same side of the split. The unit of splitting is the connected component over shared `design_id` and shared `lookalike_group`. See `data/metadata_schema.md`.

### 2026-09-16 Python 3.12, not 3.13 or 3.14

PyTorch publishes no wheels for 3.13 or 3.14 as of this date, confirmed by a failed install on 3.14. The venv is built on 3.12 until that changes. numpy and pandas are capped below their next majors for the same reason, new majors tend to land before the ecosystem follows.

### 2026-09-16 Max pooling over packing shots as the multi view rule

The score between a rider image and a garment is the max cosine over that garment's packing shots. It is the simplest version of CLAUDE.md section 4.5 and it is what the baseline uses. Mean pooling would punish a garment whose back view happens to be in the gallery when the rider photographed the front. The attention head over all pairs comes later and is measured against this.

### 2026-09-16 Role split: Sid owns data and evaluation, Shaurya owns models

Sid owns the capture protocol, the dataset, `metadata.csv`, the split tooling, the metrics, and every number that goes into `docs/results.md`. Shaurya owns the backbones, the training code, the loss and the hard negative sampling.

The point is that the person reporting a result is not the person who tuned the model that produced it. When a model author also owns the eval script, thresholds drift toward whatever makes the current checkpoint look good, usually without anyone intending it. Keeping the split means an improvement has to survive a measurement neither author controls.

Practical consequence: changes to `ml/src/metrics.py`, `ml/src/splits.py`, `ml/scripts/validate_metadata.py` and `docs/results.md` are Sid's call, and changes to training and backbone code are Shaurya's. Both still go through pull request review by the other.

### 2026-09-28 Physical-unit identity is the V1 return target

ReturnGuard verifies a physical `unit_id`, not just a catalogue `design_id`.
Two copies of the same design are therefore hard negatives: accepting a swapped
identical unit would defeat the use case. Training positives, validation labels,
retrieval evaluation, and deployed decision thresholds all use `unit_id`.
The API calls a low-confidence non-match `DIFFERENT_UNIT`; it reserves `RETAKE`
for a bad rider capture and `REFERENCE_INVALID` for a bad seller reference.

### 2026-09-28 Deploy only validation-calibrated, test-evaluated checkpoints

Every training epoch records validation Recall@1 and TPR@1% FPR and selects a
checkpoint with a declared retrieval metric rather than contrastive loss alone.
Calibration is validation-only, emits bootstrap uncertainty bounds, and records
whether it is deployable. The API refuses a non-deployable policy. The held-out
test evaluation runs separately with the selected checkpoint and is never used
to tune training or thresholds.

### 2026-09-28 Inference uses bounded, headless local evidence

The inference service accepts at most 8 packing and 12 rider photos, moves CPU
verification off the async event loop, and uses a runtime-only dependency set
with headless OpenCV. ORB descriptors are extracted once per image rather than
once per candidate pair. Quality measurements are normalized to a common maximum
side before thresholds are fitted from validation photos.

### 2026-09-28 ORB is a bounded V1 local-evidence baseline

ORB plus RANSAC is retained as optional, interpretable local evidence for tags,
logos, prints, and embroidery. SuperPoint/LightGlue remains the next benchmark,
not a silent dependency, because it needs a data-backed accuracy and latency
comparison first. `mean_rider_score` is the default global evidence field; the
InfoNCE temperature is 0.07 and the last two DINO blocks are trainable unless a
recorded validation experiment changes those defaults.
