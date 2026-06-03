# NLP GPT2 Project

See `HANDOFF.md` for the current experiment handoff, processed data schema, and
next training/evaluation steps.

## Data Preparation

External experiment datasets are not committed. Recreate the processed JSONL
files with:

```powershell
py -3 scripts\prepare_external_data.py
```

Check the processed dataset schema and prompt/metric bridge with:

```powershell
py -3 scripts\smoke_test_data_bridge.py
```
