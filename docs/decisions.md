# Decisions

Append only. Each entry: date, the decision, the alternatives, and why. If a decision is reversed later, add a new entry rather than editing the old one.

## Open

| decision | options | owner | needed by |
| --- | --- | --- | --- |
| App framework | Expo (recommended) vs Flutter | | week 3, before the capture app |
| Role split | who owns data plus eval, who owns models | | week 1 |
| Rider phones | which budget Androids we buy or borrow | | week 1, blocks capture |
| Image storage | Google Drive vs S3 | | week 1, blocks capture |
| QR tag anchor in V1 | vision only vs vision plus printed tag | | week 6 |

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
