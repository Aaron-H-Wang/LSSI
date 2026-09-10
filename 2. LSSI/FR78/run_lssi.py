from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
subprocess.run(
    [sys.executable, str(HERE.parent / "lssi_multimaterial.py"), "--material", "FR78",
     "--data", str(HERE / "CoreLossData_FR78.mat"), "--output", str(HERE)],
    check=True,
)
