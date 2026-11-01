# Usage

Run commands from the repository root after installing `requirements.txt`:

```bash
python scripts/train.py --config configs/fr78.yaml
python scripts/train.py --config configs/fr79.yaml
python scripts/train.py --config configs/fr95.yaml
python scripts/train.py --config configs/quick.yaml
```

Full configurations share bounds, penalties, and the training schedule. Dataset
paths, unit conversion, and output directories differ. The quick run is a pipeline
check and does not reproduce the reference table.

The LSSI reference uses four CPU threads (`train.threads: 4`) for every material.

Training overrides: `--data`, `--output-dir`, `--seed`, `--device`,
`--phase1-epochs`, `--phase2-epochs`, `--phase3-steps`, and `--quiet`.
Set the seed before model creation when using the Python API.

Outputs are `model.pt`, `equation.txt`, `summary.json`, and `history.json`.
Evaluation and prediction restore the checkpoint's data conversion and split:

```bash
python scripts/evaluate.py --checkpoint results/fr78/model.pt
python scripts/predict.py --checkpoint results/fr78/model.pt --freq 100e3 300e3 --flux 50 100
python scripts/predict.py --checkpoint results/fr95/model.pt --csv points.csv --out predictions.csv
```

Prediction CSV inputs need `f_Hz` and `B_mT`; output is `P_mW_cm3`. Load only
trusted checkpoints, since deserialization uses `weights_only=False`.
