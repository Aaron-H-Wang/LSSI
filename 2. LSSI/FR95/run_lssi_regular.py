from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
subprocess.run(
    [sys.executable, str(HERE.parent / "lssi_multimaterial.py"), "--material", "FR95",
     "--data", str(HERE / "CoreLossData_FR95.mat"), "--output", str(HERE / "regular_analysis")],
    check=True,
)
