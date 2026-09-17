# ml

Notebooks are for exploration. Anything that produces a reported number is a script under `ml/scripts/` with a fixed seed, and its result goes in `docs/results.md` with the commit hash.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt

python ml/scripts/make_splits.py --seed 13
python ml/scripts/validate_metadata.py
```

`validate_metadata.py` is the gate. It fails if a design or lookalike group straddles two splits, which is the failure mode that makes every downstream number meaningless.
