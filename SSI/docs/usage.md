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
metrics may differ from the paper references described below.

Training accepts `--data`, `--output-dir`, `--seed`, `--device`,
`--phase2-epochs`, and `--quiet`. Outputs are `model.pt`, `equation.txt`,
`summary.json`, and `history.json`. Checkpoints store the data and library
configurations, including reduced quick-run libraries.

Prediction uses Hz and mT, and returns mW/cm³. CSV inputs need `f_Hz` and `B_mT`;
use `--out predictions.csv` to save outputs. Load only trusted checkpoints;
deserialization uses `weights_only=False`.

## Inspect the saved paper results

The README's table and images refer to the [paper reference archive](../reference_results/).
It comes from the original experiments with fixed-support continuation, seed 0,
one CPU thread, and PyTorch 2.14.0+cpu. Standard commands use the packaged trainer
without this continuation. Increasing `--phase2-epochs` changes discovery and
pruning; it is not equivalent to continuing fixed support.

Inspect metrics, full-precision coefficients, equations, prediction CSVs, and
NumPy histories without training or loading a checkpoint:

```python
import json
from pathlib import Path
import numpy as np

reference = Path("reference_results")
metrics = json.loads((reference / "FR79.json").read_text())
print(metrics["test"]["mape_percent"])
print(metrics["active_terms"])

history = np.load(reference / "FR79_history.npz")
visible = history["coefficient_epochs"] <= 40000
epochs = history["coefficient_epochs"][visible]
coefficients = history["coefficients"][visible]
```

History coefficients are column-scaled; JSON equation coefficients have this
scaling removed. CSVs retain train/test assignments and use kHz, mT, and mW/cm³.
