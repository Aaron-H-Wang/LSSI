# SSI usage

Run from `LSSI/SSI`:

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/fr78.yaml
python scripts/train.py --config configs/fr79.yaml
python scripts/train.py --config configs/fr95.yaml
python scripts/train.py --config configs/quick.yaml
python scripts/evaluate.py --checkpoint results/fr79/model.pt
python scripts/predict.py --checkpoint results/fr79/model.pt --freq 100e3 300e3 --flux 50 100
```

The quick configuration uses a smaller library and shorter schedule. All full
configurations share a 103-term library and optimization settings.

The packaged configurations use `train.threads: 12` for all three materials. Fixed-grid support
can be sensitive to floating-point reduction order; keep this setting when
comparing package runs. Zero requests PyTorch's default. Archived and rerun
metrics are listed separately in the repository-level reference results.

Training accepts `--data`, `--output-dir`, `--seed`, `--device`,
`--phase2-epochs`, and `--quiet`. Outputs are `model.pt`, `equation.txt`,
`summary.json`, and `history.json`. Checkpoints store the data and library
configurations, including reduced quick-run libraries.

Prediction uses Hz and mT, and returns mW/cm³. CSV inputs need `f_Hz` and `B_mT`;
use `--out predictions.csv` to save outputs. Load only trusted checkpoints;
deserialization uses `weights_only=False`.
