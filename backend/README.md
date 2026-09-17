# backend

FastAPI inference server. Not started yet, it comes after the matching model has real numbers behind it. Milestone order is in CLAUDE.md section 9.

Planned surface: one endpoint that takes the packing image set and the rider image set for an order and returns MATCH, DIFFERENT_PRODUCT, SUSPICIOUS, or RETAKE, plus a score and the per check breakdown.

No secrets in code. Configuration comes from environment variables, see `.env.example` when it exists.
