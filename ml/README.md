# ml

Notebooks are for exploration. Anything that produces a reported number is a script under `ml/scripts/` with a fixed seed, and its result goes in `docs/results.md` with the commit hash.

## Setup

Python 3.12. PyTorch publishes no wheels for 3.13 or 3.14 yet, so a newer interpreter fails at install time.

```bash
brew install python@3.12
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt
```

## Layout

```
src/embed.py      backbone loading, returns L2 normalised vectors
src/metric_model.py DINOv2 retrieval model and projection head
src/training_data.py paired examples and hard-negative batch sampler
src/losses.py     symmetric cross-domain InfoNCE loss
src/fine_tuned_embed.py checkpoint-backed embedding interface for inference
src/quality.py    blur, brightness, and resolution capture-quality checks
src/set_matching.py multi-view packing-versus-rider evidence scores
src/local_features.py ORB plus RANSAC local-detail evidence matching
src/metrics.py    TPR at 1% FPR, ROC AUC, Recall@k
src/splits.py     connected components over design_id and lookalike_group
scripts/          anything that produces a reported number
tests/            fast, no torch, no images
```

## Usage

```bash
# assign splits, then check for leakage. run both after every capture session.
python ml/scripts/make_splits.py --seed 13
python ml/scripts/validate_metadata.py

# zero shot baseline. --image-root points at the image store, never inside the repo.
python ml/scripts/eval_zero_shot.py --image-root ~/returnguard-images --backbone dinov2_base --split all
python ml/scripts/eval_zero_shot.py --image-root ~/returnguard-images --backbone clip_vit_b32 --split all

pytest ml/tests -q
```

While the dataset is under about 60 garments, use `--split all` and treat the result as a sanity reading, not a headline. Switch to `--split test` once there is enough data for the split to mean anything.

## What the baseline does

Rider shots are queries, packing shots are the gallery. The score between a rider image and a garment is the max cosine over that garment's packing shots. No segmentation, no fine tuning, no fusion. It exists to be beaten, and it stays in `docs/results.md` forever as the reference.

`validate_metadata.py` is the gate before any evaluation. It fails if a design or lookalike group straddles two splits, which is the failure that would make every downstream number meaningless.

## Fine tuning the retrieval model

Run metadata validation and the frozen baselines first. Training uses one
packing photo and one rider photo for each physical garment in a batch. It
learns a shared embedding with symmetric InfoNCE: the diagonal is the positive
pair and all off-diagonal entries are negatives. ReturnGuard's current identity
policy is explicit: `--identity-policy product` (the default) groups positives
by `design_id`; `--identity-policy unit` groups positives by the globally unique
physical `garment_id`. `unit_id` is only an ordinal within a design and is never
used as a global identity. Unit training treats identical designs as hard
negatives, while product training uses only lookalikes as hard negatives.

```bash
python ml/scripts/validate_metadata.py
python ml/scripts/train_metric.py \
  --image-root /absolute/path/to/returnguard-images \
  --backbone dinov2_base --epochs 20 --batch-size 16 --identity-policy product
```

The script uses only the train split for optimisation. Each epoch reports
validation Recall@1 and TPR@1% FPR, and selects the checkpoint on validation
Recall@1 by default (use `--selection-metric tpr_at_1pct_fpr` when that is the
agreed headline). It never fits thresholds or emits test metrics. Checkpoints
go under `ml/checkpoints/`, which is ignored by git. The evaluation owner then
runs the exact same held-out protocol against the selected checkpoint:

```bash
python ml/scripts/eval_zero_shot.py \
  --image-root /absolute/path/to/returnguard-images \
  --checkpoint ml/checkpoints/best.pt --identity-policy product --split test
```

## Capture-quality gate

Run the deterministic quality checks over rider photos before embedding. The
defaults are intentionally provisional. Fit them on validation captures before
they become product thresholds, then keep the selected values fixed for a run.

```bash
python ml/scripts/check_quality.py \
  --image-root /absolute/path/to/returnguard-images \
  --shot-type rider --out ml/outputs/rider-quality.json
```

The gate marks a photo for retake when it is undersized, too dark, too bright,
or has low Laplacian variance. It does not claim that a garment is present;
that requires the planned segmentation stage.

Fit and save the real runtime thresholds from the first 30 captured units;
their sharpness and brightness are measured after normalization to a fixed
maximum side. The API requires this artifact instead of silently using guessed
defaults.

```bash
python ml/scripts/check_quality.py \
  --image-root /absolute/path/to/returnguard-images --shot-type all \
  --fit-first-units 30 --fit-out ml/checkpoints/quality-policy.json
```

## Multi-view evidence

`src/set_matching.py` compares every rider embedding with every packing
embedding. It returns the existing max-pair score plus the average and lower
quartile of each rider image's best packing match. The latter two tell a future
calibration step whether one clear photo is masking several contradictory ones.
The module emits scores and the strongest image-pair indices only. It never
assigns a return decision without validation-fitted thresholds.

## Local-detail evidence

For tags, logos, embroidery, and rigid print patches, use the ORB plus RANSAC
baseline to find geometrically consistent local matches. It is most useful when
the global embedding is uncertain, not on uniform or heavily occluded fabric.

```bash
python ml/scripts/match_local_features.py \
  --packing /path/to/packing-01.jpg /path/to/packing-02.jpg \
  --rider /path/to/rider-01.jpg /path/to/rider-02.jpg
```

The output reports the strongest packing/rider pair, the number of feature
matches that survived the ratio test, and the RANSAC geometric-inlier count.
Those are evidence values for later validation calibration, not a return verdict.

## Calibrate deployment decisions

After training, fit a decision policy on validation data only. The script
compares every rider shot to every packing set in the validation split, creating
several deploy-shaped trials per identity before it fits strict MATCH and
DIFFERENT_PRODUCT/DIFFERENT_UNIT thresholds with an explicit SUSPICIOUS band.
Choose the same identity policy used for training and held-out evaluation. The
script writes a bootstrap uncertainty report beside the policy and never touches
the test split.

It always writes the calibration artifact and marks it `is_deployable: false`
when the validation sample is too small or discrimination is weak. The API
refuses to load such an artifact; this preserves the diagnostic record without
allowing accidental deployment. By default, calibration requires at least 40
positive and 100 negative validation comparisons.

```bash
python ml/scripts/calibrate_decisions.py \
  --image-root /absolute/path/to/returnguard-images \
  --checkpoint ml/checkpoints/best.pt \
  --identity-policy product --out ml/checkpoints/policy.json
```

The resulting `policy.json` must ship with the exact checkpoint that produced
it. Do not fit or adjust it using test-set comparisons.
