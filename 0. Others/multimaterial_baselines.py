"""Train SE, random-forest, and feedforward-NN baselines on all materials.

The script uses the same lexicographic sorting and every-fifth-point test split
as SSI and LSSI. Models operate on log10(f), log10(B), and log10(P), while all
reported MAPE and R2 values are evaluated after transforming predictions back
to physical core-loss units.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ("FR78", "FR79", "FR95")


def load(material):
    path = ROOT / "2. LSSI" / material / f"CoreLossData_{material}.mat"
    data = scipy.io.loadmat(path)
    f = np.asarray(data["f"], dtype=float).ravel()
    b = np.asarray(data["B"], dtype=float).ravel()
    p = np.asarray(data["P"], dtype=float).ravel()
    if material == "FR95":
        f /= 1000.0
        b /= 10.0
    order = np.lexsort((b, f))
    return f[order], b[order], p[order]


def metrics(y, prediction):
    residual = prediction - y
    return {
        "mape_percent": float(np.mean(np.abs(residual / y)) * 100.0),
        "r2": float(r2_score(y, prediction)),
        "rmse_mw_cm3": float(np.sqrt(np.mean(residual**2))),
        "mean_residual_percent": float(np.mean(residual / y) * 100.0),
        "std_residual_percent": float(np.std(residual / y) * 100.0),
    }


def parameter_count(model, name):
    if name == "SE":
        return 3
    if name == "RF":
        # Number of learned split/leaf nodes, a transparent reproducible proxy
        # for the forest's stored numerical structure.
        return int(sum(tree.tree_.node_count for tree in model.estimators_))
    if name == "FNN":
        network = model.named_steps["mlpregressor"]
        return int(sum(w.size for w in network.coefs_) + sum(b.size for b in network.intercepts_))
    raise KeyError(name)


def train_material(material, seed, output):
    f, b, p = load(material)
    test = np.zeros(len(p), dtype=bool)
    test[::5] = True
    x = np.column_stack((np.log10(f), np.log10(b)))
    y = np.log10(p)

    models = {
        "SE": LinearRegression(),
        "RF": RandomForestRegressor(n_estimators=200, min_samples_leaf=1,
                                    random_state=seed, n_jobs=-1),
        "FNN": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", solver="lbfgs",
                         max_iter=5000, random_state=seed, tol=1e-9),
        ),
    }
    result = {"material": material, "seed": seed, "split": "sorted every fifth point"}
    predictions = {}
    for name, model in models.items():
        model.fit(x[~test], y[~test])
        predicted = 10.0 ** model.predict(x[test])
        result[name] = metrics(p[test], predicted)
        result[name]["learned_parameter_count"] = parameter_count(model, name)
        predictions[name] = predicted.tolist()
    result["test_data"] = {
        "frequency_khz": f[test].tolist(), "flux_density_mt": b[test].tolist(),
        "measured_mw_cm3": p[test].tolist(), "predictions_mw_cm3": predictions,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{material}_baselines.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "0. Others" / "multimaterial_results")
    args = parser.parse_args()
    results = [train_material(material, args.seed, args.output) for material in MATERIALS]
    print(json.dumps({item["material"]: {name: item[name] for name in ("SE", "RF", "FNN")}
                      for item in results}, indent=2))


if __name__ == "__main__":
    main()
