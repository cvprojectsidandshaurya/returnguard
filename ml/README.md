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
pair and all off-diagonal entries are negatives. The sampler keeps garment IDs
unique within a batch and, whenever the data permits, places an identical
design or lookalike in the same batch as a hard negative.

```bash
python ml/scripts/validate_metadata.py
python ml/scripts/train_metric.py \
  --image-root /absolute/path/to/returnguard-images \
  --backbone dinov2_base --epochs 20 --batch-size 16
```

The script uses only the train split for optimisation and validation loss to
select a checkpoint. It never fits thresholds or emits test metrics. Checkpoints
go under `ml/checkpoints/`, which is ignored by git. Once a checkpoint is
selected, the evaluation owner should run the held-out retrieval and
verification evaluation with thresholds fitted on validation data only.

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

After training, fit the decision policy on validation garments only. The script
compares every rider set to every packing set in the validation split, then
creates a strict MATCH threshold, a strict DIFFERENT_PRODUCT threshold, and an
explicit SUSPICIOUS band between them.

It refuses to write a deployment policy if either strict side recognizes fewer
than half of its own validation examples. That is a safety stop, not a result to
work around by lowering the threshold.

```bash
python ml/scripts/calibrate_decisions.py \
  --image-root /absolute/path/to/returnguard-images \
  --checkpoint ml/checkpoints/best.pt \
  --out ml/checkpoints/policy.json
```

The resulting `policy.json` must ship with the exact checkpoint that produced
it. Do not fit or adjust it using test-set comparisons.
