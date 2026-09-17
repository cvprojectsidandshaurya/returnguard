# ml

Notebooks are for exploration. Anything that produces a reported number is a script under `ml/scripts/` with a fixed seed, and its result goes in `docs/results.md` with the commit hash.

## Setup

Python 3.12. PyTorch publishes no wheels for 3.13 or 3.14 yet, so a newer interpreter fails at install time.

```bash
brew install python@3.12
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt
```

## Layout

```
src/embed.py      backbone loading, returns L2 normalised vectors
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
