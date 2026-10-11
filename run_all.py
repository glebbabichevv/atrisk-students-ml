"""Reproduce every table, figure and number in the paper.

    python run_all.py            # about 15 minutes on a 2-core laptop

Steps: cross-validation experiments -> secondary analyses and figures ->
LaTeX tables and number macros (paper/generated/).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for script in ["experiments.py", "analysis.py", "make_tables.py"]:
    print(f"\n=== {script} ===", flush=True)
    subprocess.run([sys.executable, str(ROOT / "src" / script)], check=True)
print("\nDone. Build the paper with:  cd paper && latexmk -pdf main.tex")
