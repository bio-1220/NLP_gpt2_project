# NLP GPT-2 Project

CS224n-style GPT-2 project with emotion-aware poem generation experiments.

## Project Docs

```text
proposal.md  -> project proposal and research motivation
HANDOFF.md   -> current data, experiment, and GPU-run handoff
```

## Data

Raw and processed external datasets are not committed.

Recreate the base processed data:

```powershell
py -3 scripts\prepare_external_data.py
```

Check the processed data bridge:

```powershell
py -3 scripts\smoke_test_data_bridge.py
```

For diagnostic experiment data, see `HANDOFF.md`.

