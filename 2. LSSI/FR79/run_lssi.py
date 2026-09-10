from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
subprocess.run(
    [sys.executable, str(HERE.parent / "lssi_multimaterial.py"), "--material", "FR79",
     "--data", str(HERE / "CoreLossData_FR79.mat"), "--output", str(HERE)],
    check=True,
)
