"""Convenience runner; every plot module remains independently executable."""

import importlib


MODULES = (
    "plot_01_measurement_agreement",
    "plot_02_operating_domains",
    "plot_03_lssi_validation",
    "plot_04_residual_histograms",
    "plot_05_residual_vs_flux",
    "plot_06_residual_vs_frequency",
    "plot_05_method_comparison",
    "plot_06_radar_comparison",
    "plot_07_sparse_coefficient_evolution",
    "plot_08_raw_parameter_evolution",
    "plot_07_common_structure",
)


if __name__ == "__main__":
    for module_name in MODULES:
        print(f"Generating {module_name} ...")
        importlib.import_module(module_name).main()
