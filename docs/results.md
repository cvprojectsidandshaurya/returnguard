# Results

One row per experiment. Numbers that are not in this table do not exist. Every row comes from a script run with a fixed seed, at a recorded commit.

Headline metrics are TPR at 1% FPR (verification) and Recall@1 (retrieval). Never report accuracy alone. Zero shot baselines stay in the table forever, they are the reference every later model is judged against.

| date | commit | data version | model | config | TPR@1%FPR | R@1 | R@5 | ECE | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | | |

Per condition breakdowns live in `docs/results/` as one markdown file per experiment, named `YYYY-MM-DD-<short-commit>-<model>.md`. Each one carries the polybag vs not, dim vs good light, crumpled vs folded, and easy vs identical unit negative splits.
