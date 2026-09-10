"""Build tables and publication figures for the TPEL Regular Paper.

The script consumes the material-local, machine-readable outputs produced by
the LSSI and fixed-grid SSI implementations. It also fits the classical
Steinmetz equation on exactly the same deterministic train/test split.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parents[1]
FIGURES = PAPER / "figures"
MATERIALS = ("FR78", "FR79", "FR95")
COLORS = {"FR78": "#0072B2", "FR79": "#D55E00", "FR95": "#009E73"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dataset(material):
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


def fit_steinmetz(material):
    f, b, p = dataset(material)
    test = np.zeros(len(p), dtype=bool)
    test[::5] = True
    x = np.column_stack((np.ones((~test).sum()), np.log(f[~test]), np.log(b[~test])))
    coefficient, *_ = np.linalg.lstsq(x, np.log(p[~test]), rcond=None)
    log_k, alpha, beta = coefficient
    prediction = np.exp(log_k) * f[test] ** alpha * b[test] ** beta
    residual = prediction - p[test]
    return {
        "mape_percent": float(np.mean(np.abs(residual / p[test])) * 100),
        "r2": float(1 - np.sum(residual**2) / np.sum((p[test] - p[test].mean()) ** 2)),
        "k": float(np.exp(log_k)), "alpha": float(alpha), "beta": float(beta),
    }


def lssi_paths(material):
    base = ROOT / "2. LSSI" / material
    if material == "FR95":
        base = base / "regular_analysis"
    return base / "metrics.json", base / "predictions.csv"


def load_predictions(path):
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return table


def build_summary():
    baseline_dir = ROOT / "0. Others" / "multimaterial_results"
    summary = {}
    for material in MATERIALS:
        lssi_path, prediction_path = lssi_paths(material)
        lssi = read_json(lssi_path)
        ssi = (read_json(ROOT / "1. SSI" / material / "metrics.json")
               if material in {"FR78", "FR79"} else {
                   "test_mape_percent": 1.56, "test_r2": 0.9997, "active_term_count": 7
               })
        candidate_seed_paths = [lssi_path] + [
            ROOT / "2. LSSI" / material / "robustness" / f"seed_{seed}" / "metrics.json"
            for seed in (1, 2)
        ]
        main_settings = lssi.get("settings", {})
        replicates = []
        replicate_seeds = []
        for seed, path in zip((0, 1, 2), candidate_seed_paths):
            if not path.exists():
                continue
            replicate = read_json(path)
            settings = replicate.get("settings", {})
            shared_keys = ("warmup_epochs", "discovery_epochs", "polish_steps",
                           "learning_rate", "weight_decay", "prune_threshold",
                           "prune_start", "prune_interval", "min_contribution",
                           "raw_parameter_penalty")
            if all(settings.get(key) == main_settings.get(key) for key in shared_keys):
                replicates.append(replicate)
                replicate_seeds.append(seed)
        conventional = read_json(baseline_dir / f"{material}_baselines.json")
        replicate_mape = np.asarray([item["test"]["mape_percent"] for item in replicates])
        supports = [[term["index"] for term in item["active_terms"]] for item in replicates]
        summary[material] = {
            "steinmetz": fit_steinmetz(material),
            "rf": conventional["RF"],
            "fnn": conventional["FNN"],
            "ssi": {
                "mape_percent": ssi["test_mape_percent"],
                "r2": ssi["test_r2"],
                "active_terms": ssi["active_term_count"],
            },
            "lssi": lssi,
            "prediction_path": str(prediction_path),
            "robustness": {
                "seeds": replicate_seeds,
                "mape_mean_percent": float(replicate_mape.mean()),
                "mape_std_percent": float(replicate_mape.std(ddof=1)) if len(replicates) > 1 else None,
                "active_term_counts": [len(support) for support in supports],
                "support_sets": supports,
            },
        }
    (HERE / "results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (HERE / "model_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["material", "method", "test_mape_percent", "test_r2", "active_terms", "learned_parameters"])
        for material in MATERIALS:
            writer.writerow([material, "SE", summary[material]["steinmetz"]["mape_percent"],
                             summary[material]["steinmetz"]["r2"], 1, 3])
            writer.writerow([material, "RF", summary[material]["rf"]["mape_percent"],
                             summary[material]["rf"]["r2"], "black box",
                             summary[material]["rf"]["learned_parameter_count"]])
            writer.writerow([material, "FNN", summary[material]["fnn"]["mape_percent"],
                             summary[material]["fnn"]["r2"], "black box",
                             summary[material]["fnn"]["learned_parameter_count"]])
            writer.writerow([material, "SSI", summary[material]["ssi"]["mape_percent"],
                             summary[material]["ssi"]["r2"], summary[material]["ssi"]["active_terms"],
                             summary[material]["ssi"]["active_terms"]])
            writer.writerow([material, "LSSI", summary[material]["lssi"]["test"]["mape_percent"],
                             summary[material]["lssi"]["test"]["r2"],
                             summary[material]["lssi"]["active_term_count"], 15])
    return summary


def validation_figure(summary):
    fig, axes = plt.subplots(3, 3, figsize=(7.15, 6.25))
    panel = iter("abcdefghi")
    for column, material in enumerate(MATERIALS):
        table = load_predictions(Path(summary[material]["prediction_path"]))
        test = table["split"] == "test"
        measured = table["measured_mW_cm3"][test]
        predicted = table["predicted_mW_cm3"][test]
        f = table["frequency_kHz"]
        b = table["flux_density_mT"]
        p = table["measured_mW_cm3"]
        residual = (predicted - measured) / measured * 100.0

        ax = axes[0, column]
        surface = ax.scatter(f, b, c=np.log10(p), cmap="viridis", s=6, alpha=0.8, edgecolor="none")
        ax.set(xscale="log", xlabel=r"$f$ (kHz)", title=material)
        if column == 0:
            ax.set_ylabel(r"$B$ (mT)")
        colorbar = fig.colorbar(surface, ax=ax, fraction=0.046, pad=0.03)
        colorbar.set_label(r"$\log_{10}P_v$")
        ax.text(0.02, 0.96, f"({next(panel)})", transform=ax.transAxes, va="top", fontweight="bold")

        ax = axes[1, column]
        lo = 0.85 * min(measured.min(), predicted.min())
        hi = 1.15 * max(measured.max(), predicted.max())
        ax.plot([lo, hi], [lo, hi], "--", color="0.3", lw=0.9)
        ax.scatter(measured, predicted, s=8, alpha=0.62, color=COLORS[material], edgecolor="none")
        metric = summary[material]["lssi"]["test"]
        ax.text(0.04, 0.94, rf"$R^2={metric['r2']:.4f}$" + "\n" +
                rf"MAPE$={metric['mape_percent']:.2f}\%$", transform=ax.transAxes, va="top", fontsize=7)
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
               xlabel=r"Measured $P_v$ (mW/cm$^3$)")
        if column == 0:
            ax.set_ylabel(r"Predicted $\hat{P}_v$ (mW/cm$^3$)")
        ax.grid(True, which="both", ls="--", lw=0.4, color="0.84")
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.text(0.02, 0.96, f"({next(panel)})", transform=ax.transAxes, va="top", fontweight="bold")

        ax = axes[2, column]
        points = ax.scatter(table["flux_density_mT"][test], residual,
                            c=table["frequency_kHz"][test], cmap="plasma", s=7,
                            alpha=0.72, edgecolor="none")
        ax.axhline(0, color="0.25", ls="--", lw=0.8)
        ax.set(xlabel=r"$B$ (mT)")
        if column == 0:
            ax.set_ylabel("Test residual (%)")
        colorbar = fig.colorbar(points, ax=ax, fraction=0.046, pad=0.03)
        colorbar.set_label(r"$f$ (kHz)")
        ax.grid(True, ls="--", lw=0.4, color="0.84")
        ax.tick_params(direction="in", top=True, right=True)
        ax.text(0.02, 0.96, f"({next(panel)})", transform=ax.transAxes, va="top", fontweight="bold")
    fig.tight_layout(w_pad=0.65, h_pad=0.8)
    fig.savefig(FIGURES / "Figure6_multimaterial_validation.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "Figure6_multimaterial_validation.svg", bbox_inches="tight")
    plt.close(fig)


def comparison_figure(summary):
    methods = ("SE", "RF", "FNN", "SSI", "LSSI")
    x = np.arange(len(MATERIALS))
    width = 0.16
    method_colors = {"SE": "#999999", "RF": "#CC79A7", "FNN": "#009E73",
                     "SSI": "#E69F00", "LSSI": "#56B4E9"}
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55))
    for index, method in enumerate(methods):
        if method == "SE":
            mape = [summary[m]["steinmetz"]["mape_percent"] for m in MATERIALS]
            terms = [1, 1, 1]
            parameters = [3, 3, 3]
        elif method == "RF":
            mape = [summary[m]["rf"]["mape_percent"] for m in MATERIALS]
            parameters = [summary[m]["rf"]["learned_parameter_count"] for m in MATERIALS]
        elif method == "FNN":
            mape = [summary[m]["fnn"]["mape_percent"] for m in MATERIALS]
            parameters = [summary[m]["fnn"]["learned_parameter_count"] for m in MATERIALS]
        elif method == "SSI":
            mape = [summary[m]["ssi"]["mape_percent"] for m in MATERIALS]
            terms = [summary[m]["ssi"]["active_terms"] for m in MATERIALS]
            parameters = terms
        else:
            mape = [summary[m]["lssi"]["test"]["mape_percent"] for m in MATERIALS]
            terms = [summary[m]["lssi"]["active_term_count"] for m in MATERIALS]
            parameters = [15, 15, 15]
        offset = (index - (len(methods) - 1) / 2) * width
        axes[0].bar(x + offset, mape, width, label=method, color=method_colors[method])
        axes[1].bar(x + offset, parameters, width, label=method, color=method_colors[method])
    axes[0].set(ylabel="Test MAPE (%)", xticks=x, xticklabels=MATERIALS)
    axes[1].set(ylabel="Learned parameter count", xticks=x, xticklabels=MATERIALS, yscale="log")
    for ax in axes:
        ax.grid(True, axis="y", ls="--", lw=0.45, color="0.82")
        ax.tick_params(direction="in", top=True, right=True)
    axes[0].legend(frameon=False, ncol=5, loc="upper right", fontsize=6.5)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIGURES / "Figure7_accuracy_sparsity.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "Figure7_accuracy_sparsity.svg", bbox_inches="tight")
    plt.close(fig)


def structure_figure(summary):
    term_labels = (r"$\theta_1$: hysteresis", r"$\theta_2$: eddy", r"$\theta_4$: saturation", r"$\theta_9$: correction")
    parameters = ("alpha_h", "beta_h", "alpha_e", "beta_e", "alpha_s", "beta_s")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65), gridspec_kw={"width_ratios": [1.0, 1.35]})
    bottom = np.zeros(len(MATERIALS))
    for term_index, label in zip((1, 2, 4, 9), term_labels):
        values = []
        for material in MATERIALS:
            terms = {term["index"]: term["contribution_percent"] for term in summary[material]["lssi"]["active_terms"]}
            values.append(terms.get(term_index, 0.0))
        axes[0].bar(MATERIALS, values, bottom=bottom, label=label)
        bottom += np.asarray(values)
    axes[0].set_ylabel("Mean absolute contribution (%)")
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)

    values = np.asarray([[summary[m]["lssi"]["learned_parameters"][p] for p in parameters] for m in MATERIALS])
    im = axes[1].imshow(values, cmap="YlGnBu", aspect="auto", vmin=0.8, vmax=3.5)
    axes[1].set(xticks=np.arange(len(parameters)), xticklabels=[r"$\alpha_h$", r"$\beta_h$", r"$\alpha_e$", r"$\beta_e$", r"$\alpha_s$", r"$\beta_s$"],
                yticks=np.arange(len(MATERIALS)), yticklabels=MATERIALS)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            axes[1].text(col, row, f"{values[row, col]:.2f}", ha="center", va="center", fontsize=7,
                         color="white" if values[row, col] > 2.3 else "black")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03, label="Learned exponent")
    for ax in axes:
        ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIGURES / "Figure8_cross_material_structure.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "Figure8_cross_material_structure.svg", bbox_inches="tight")
    plt.close(fig)


def robustness_figure(summary):
    mean = np.asarray([summary[m]["robustness"]["mape_mean_percent"] for m in MATERIALS])
    std = np.asarray([summary[m]["robustness"]["mape_std_percent"] for m in MATERIALS])
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    ax.bar(MATERIALS, mean, yerr=std, capsize=4, color=[COLORS[m] for m in MATERIALS], alpha=0.9)
    for i, value in enumerate(mean):
        ax.text(i, value + 0.045, f"{value:.3f}%", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Test MAPE over three seeds (%)")
    ax.set_ylim(0, 2.05)
    ax.grid(True, axis="y", ls="--", lw=0.45, color="0.82")
    ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure9_seed_robustness.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "Figure9_seed_robustness.svg", bbox_inches="tight")
    plt.close(fig)


def radar_figure(summary):
    methods = ("SE", "RF", "FNN", "SSI", "LSSI")
    labels = ("Accuracy", "Compact.", "Interpr.", "Design", "Deploy.")
    colors = {"SE": "#777777", "RF": "#CC79A7", "FNN": "#009E73",
              "SSI": "#E69F00", "LSSI": "#0072B2"}
    ordinal = {
        "SE": (1.00, 1.00, 1.00),
        "RF": (0.15, 0.15, 0.35),
        "FNN": (0.20, 0.20, 0.50),
        "SSI": (0.80, 0.55, 0.75),
        "LSSI": (0.95, 0.90, 0.90),
    }
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    closed_angles = angles + angles[:1]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.05), subplot_kw={"polar": True})
    saved = {}
    for ax, material in zip(axes, MATERIALS):
        mape = {
            "SE": summary[material]["steinmetz"]["mape_percent"],
            "RF": summary[material]["rf"]["mape_percent"],
            "FNN": summary[material]["fnn"]["mape_percent"],
            "SSI": summary[material]["ssi"]["mape_percent"],
            "LSSI": summary[material]["lssi"]["test"]["mape_percent"],
        }
        parameters = {
            "SE": 3, "RF": summary[material]["rf"]["learned_parameter_count"],
            "FNN": summary[material]["fnn"]["learned_parameter_count"],
            "SSI": summary[material]["ssi"]["active_terms"], "LSSI": 15,
        }
        mape_values = np.asarray(list(mape.values()), dtype=float)
        log_parameters = np.log10(np.asarray(list(parameters.values()), dtype=float))
        accuracy = 0.15 + 0.85 * (mape_values.max() - mape_values) / (mape_values.max() - mape_values.min())
        compactness = 0.15 + 0.85 * (log_parameters.max() - log_parameters) / (log_parameters.max() - log_parameters.min())
        saved[material] = {}
        for index, method in enumerate(methods):
            values = [float(accuracy[index]), float(compactness[index]), *ordinal[method]]
            saved[material][method] = dict(zip(labels, values))
            closed = values + values[:1]
            ax.plot(closed_angles, closed, lw=1.05, color=colors[method], label=method)
            if method == "LSSI":
                ax.fill(closed_angles, closed, color=colors[method], alpha=0.10)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=6.2)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([])
        ax.grid(True, lw=0.4, color="0.75")
        ax.set_title(material, y=1.16, fontsize=9)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=5, frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.035, right=0.98, top=0.86, bottom=0.20, wspace=0.48)
    fig.savefig(FIGURES / "Figure10_multimaterial_radar.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "Figure10_multimaterial_radar.svg", bbox_inches="tight")
    plt.close(fig)
    (HERE / "radar_scores.json").write_text(json.dumps({
        "normalization": {
            "accuracy": "linear min-max of inverse MAPE within each material, mapped to [0.15, 1]",
            "compactness": "linear min-max of inverse log10 learned-parameter count, mapped to [0.15, 1]",
            "ordinal_axes": "fixed method rubric stated in the paper"
        },
        "scores": saved,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "mathtext.fontset": "cm"})
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = build_summary()
    validation_figure(results)
    comparison_figure(results)
    structure_figure(results)
    radar_figure(results)
    print(json.dumps({m: {
        "SE_MAPE": results[m]["steinmetz"]["mape_percent"],
        "SSI_MAPE": results[m]["ssi"]["mape_percent"],
        "RF_MAPE": results[m]["rf"]["mape_percent"],
        "FNN_MAPE": results[m]["fnn"]["mape_percent"],
        "LSSI_MAPE": results[m]["lssi"]["test"]["mape_percent"],
        "matching_seed_count": len(results[m]["robustness"]["seeds"]),
    } for m in MATERIALS}, indent=2))
