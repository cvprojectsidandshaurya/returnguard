# Status and next steps

Last updated 2026-09-16. Keep this current. It is the first thing a new person or a coding agent should read after `CLAUDE.md`.

## Done

- Both members are org owners. GitHub org and public repo, `main` protected by the `protect-main` ruleset: pull request required, one approving review, no force push, no deletion. Nobody pushes to main, owners included.
- Monorepo scaffolded: `ml`, `data`, `backend`, `docs`. No `/app`, the Expo vs Flutter decision is still open.
- Capture protocol v1.0 frozen in `data/capture_protocol.md`. Shot lists, conditions, naming scheme, definition of done per garment.
- Metadata schema in `data/metadata_schema.md`. `data/metadata.csv` and `data/phones.csv` exist with headers only.
- Split tooling. `ml/scripts/make_splits.py` assigns train/val/test by split group with a fixed seed. `ml/scripts/validate_metadata.py` fails loudly on leakage.
- Zero shot baseline in PR 1, awaiting review. DINOv2 and CLIP backbones, TPR at 1% FPR and Recall@k, per condition breakdown, 9 passing tests.

Nothing has been photographed yet. `metadata.csv` is empty. No number in `docs/results.md` yet.

## Who owns what

Sid owns data and evaluation: the capture protocol, the dataset and `metadata.csv`, the split tooling, the metrics, and every number in `docs/results.md`.
Shaurya owns models: backbones, training code, losses, and hard negative sampling.

The person reporting a result is never the person who tuned the model that produced it. See `docs/decisions.md`.

## Left to do, in order

1. Review and merge PR 1.
2. Decide the image store, Google Drive or S3, create it, record the path in `docs/decisions.md`.
3. Install Python 3.12 and build the venv. PyTorch has no wheels for 3.13 or 3.14.
4. Photograph the first 30 garments against the capture protocol, filling `metadata.csv` in the same session.
5. Run `make_splits.py`, then `validate_metadata.py`, then `eval_zero_shot.py` on both DINOv2 and CLIP. Add the row to `docs/results.md`.

That last step is the first real milestone. Everything after it is in `CLAUDE.md` section 9.

## Decided since the plan was written

- Garments first, not shoes. Shoes are rigid, so matching them is close to solved and the research contribution disappears. Footwear can come later as a contrast condition, with its own protocol version.
- The first batch is shot on iPhones as a pipeline check. An iPhone can reproduce crumpling, polybag glare, dim light, blur and partial views, but not the sensor and processing of a budget Android. Any number from an iPhone only batch is an upper bound and must say so in the notes column. Before the dataset passes about 50 garments, re-shoot the rider side of a 20 garment subset on a cheap Android. The gap between the two is a result, not an inconvenience.
- Review photos scraped from marketplaces are not the training set. Use the licensed academic equivalents (DeepFashion2 consumer to shop, Street2Shop) for pretraining. Scraping breaks marketplace terms and the photos carry other people's faces and homes. Review photos also lack the polybag and crumple conditions and carry no unit level labels, so they cannot serve as evaluation data.

## How the model is trained, in short

There is no real versus fake label and the model never learns fraud. Every image is a real photo of a real garment, labelled with `garment_id`. The model learns an embedding: same garment maps to nearby vectors, different garments map apart. Fraud is inferred later from a low similarity score.

Negatives are not collected or uploaded. In a batch of B pairs, the diagonal of the similarity matrix is the B positives and every off diagonal cell is a negative, so a batch of 64 yields 4,032 negatives for free (InfoNCE). What matters is which negatives appear: random ones go stale within an epoch, so the sampler forces hard ones into every batch using `design_id` and `lookalike_group`. That is why the protocol requires buying two or three identical units of the same design.

The MATCH, SUSPICIOUS, DIFFERENT_PRODUCT, RETAKE decision is not part of training. Training produces a score. The threshold is fitted on the validation split at the point where the false positive rate reaches 1%, and the band around it becomes SUSPICIOUS.

## Rules that are easy to break by accident

- No images, weights or datasets in git, ever. Only `metadata.csv` and small config.
- Never split at the image level. The unit is the split group, the connected component over shared `design_id` and shared `lookalike_group`. `validate_metadata.py` enforces it, run it before any evaluation.
- Every reported number comes from a script with a fixed seed and a recorded commit hash, and gets one row in `docs/results.md`.
- Zero shot baselines stay in the table forever and are reported alongside every fine tuned model.
- Headline metrics are TPR at 1% FPR and Recall@1 with per condition breakdowns. Never accuracy alone.
- All work goes through a pull request reviewed by the other member.
- No em dashes in any doc or copy.

## Commands

```bash
# setup, Python 3.12 only
brew install python@3.12
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt

# after every capture session
python ml/scripts/make_splits.py --seed 13
python ml/scripts/validate_metadata.py

# the baseline
python ml/scripts/eval_zero_shot.py --image-root ~/returnguard-images --backbone dinov2_base --split all
python ml/scripts/eval_zero_shot.py --image-root ~/returnguard-images --backbone clip_vit_b32 --split all

pytest ml/tests -q
```

Use `--split all` while the dataset is under about 60 garments and treat the result as a sanity reading. Switch to `--split test` once the split means something.
