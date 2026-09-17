# data

Metadata only. No images ever land here or anywhere else in git, see `.gitignore`.

- `capture_protocol.md` the shooting rules. Read before photographing anything.
- `metadata_schema.md` columns, allowed values, and the split rules.
- `metadata.csv` one row per image.
- `phones.csv` the phones used for capture.

Images live in the shared image store. The storage choice (Google Drive vs S3) is still open, see `docs/decisions.md`.
