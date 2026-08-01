# Swedish Loghi Fine-Tuning -- Session Report

- Dataset root: `F:\huggingface_dataset`
- Run id: `None`
- Cumulative epoch: 0
- Global step: 0
- Sessions completed: 0
- Cumulative training time: 0.0s
- Best validation CER: None

## Pilot split
- Train: 9999 (target 10000)
- Val: 1000 (target 1000)
- Test reserved: 810 (target 1000)
- Shortfalls: ('train: requested 10000, achieved 9999 (largest defensible split under max_collection_fraction=0.25)', 'test_reserved: requested 1000, achieved 810')

## Character compatibility
- Lines examined: 10999
- Compatible: True
- Unsupported characters: 0

## RTX 3070 memory probe
- Chosen batch size: 16
- Peak VRAM: 2422.0 MB / 8192.0 MB total
- Safety margin: 5770.0 MB

## Full-state resume proof
- Proof passed: True
- Epoch continued (not restarted): True
- Global step continued (not restarted): True

## Continuing later

```
PYTHONPATH=src .venv/Scripts/python.exe scripts/train_loghi_swedish.py
# then choose option 6 (Start or resume five-hour session)
```

**Note:** falling training loss/CER here is evidence the training mechanics work, not a quality claim. `loghi_swedish_finetuned_v1` has not been evaluated on the reserved test set or the Lion-vs-Loghi comparison; do not interpret these numbers as benchmark quality.