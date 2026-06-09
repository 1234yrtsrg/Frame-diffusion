from pathlib import Path
import runpy
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
CSDI_DIR = REPO_ROOT / "CSDI"
sys.path.insert(0, str(CSDI_DIR))
runpy.run_path(str(CSDI_DIR / "sample_express4d.py"), run_name="__main__")
