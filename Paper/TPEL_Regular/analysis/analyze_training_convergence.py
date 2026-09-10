"""Quantify late-stage LSSI convergence for the three material runs.

The script is intentionally independent from the plotting scripts so that the
numerical convergence audit can be rerun after any training change.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MATERIALS = ("FR78", "FR79", "FR95")
COEFFICIENTS = tuple(f"coef_{index}" for index in range(1, 11))
RAW_PARAMETERS = (
    "raw_h_f", "raw_h_B", "raw_e_f", "raw_e_B", "raw_s_f", "raw_s_B",
    "gamma_h", "raw_fc_e", "raw_pc_e", "raw_fc_h", "raw_pc_h",
)


def history_path(material: str) -> Path:
    base = ROOT / "2. LSSI" / material
    if material == "FR95":
        base /= "regular_analysis"
    return base / "parameter_history.csv"


def main() -> None:
    records = []
    for material in MATERIALS:
        with history_path(material).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        joint = [row for row in rows if row["phase"] == "joint discovery"]
        terminal_epoch = int(joint[-1]["epoch"])
        audit_start = 1200 + int(0.75 * 40000)
        start = min(joint, key=lambda row: abs(int(row["epoch"]) - audit_start))
        end = joint[-1]

        coefficient_changes = {
            name: abs(float(end[name]) - float(start[name])) for name in COEFFICIENTS
        }
        raw_changes = {
            name: abs(float(end[name]) - float(start[name])) for name in RAW_PARAMETERS
        }
        records.append({
            "material": material,
            "audit_start_epoch": int(start["epoch"]),
            "audit_end_epoch": terminal_epoch,
            "max_abs_coefficient_change": max(coefficient_changes.values()),
            "max_abs_raw_coordinate_change": max(raw_changes.values()),
            "coefficient_at_max": max(coefficient_changes, key=coefficient_changes.get),
            "raw_coordinate_at_max": max(raw_changes, key=raw_changes.get),
            "coefficient_tolerance": 1.5e-4,
            "raw_coordinate_tolerance": 1.0e-3,
            "converged": (
                max(coefficient_changes.values()) <= 1.5e-4
                and max(raw_changes.values()) <= 1.0e-3
            ),
        })

    csv_path = HERE / "convergence_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    (HERE / "convergence_diagnostics.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
