"""Project paths. Every script reads and writes through these."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"
GENERATED = ROOT / "paper" / "generated"

UCI_DATA = DATA / "dataset.csv"
UPV_DATA = DATA / "upv"
UCI_RESULTS = RESULTS / "uci"
UPV_RESULTS = RESULTS / "upv"
