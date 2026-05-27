from pathlib import Path
import runpy
import sys


csdi_dir = Path(__file__).resolve().parent / "CSDI"
sys.path.insert(0, str(csdi_dir))
runpy.run_path(str(csdi_dir / "sample_express4d.py"), run_name="__main__")
