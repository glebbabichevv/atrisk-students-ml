"""Reproduce every table, figure and number in the paper.

    python run_all.py                    # every step
    python run_all.py --list             # show the steps, run nothing
    python run_all.py --only upv         # one group (uci, upv, report) or step names
    python run_all.py --skip uci.tuning  # leave out a step (tuning is the slowest)
    python run_all.py --from uci.alerts  # resume after a failed step
    python run_all.py --paper            # build paper/main.pdf at the end

Each step is a standalone script in src/. Its output is also saved to
results/logs/<step>.log. results/run_manifest.json records the commit, library
versions, data checksums and step durations of the last full run; partial runs
are recorded in results/logs/last_partial_run.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA, RES = ROOT / "data", ROOT / "results"
UCI, UPV = RES / "uci", RES / "upv"
LOGS = RES / "logs"
MANIFEST, DURATIONS = RES / "run_manifest.json", LOGS / "durations.json"

UCI_FILES = [DATA / "dataset.csv"]
UPV_FILES = [DATA / "upv" / f"dataset_{y}_hash.csv" for y in (2018, 2021, 2022)]
OOF = [UCI / f"oof_{s}.npz" for s in ("binary", "multiclass", "sem1", "enrol")]
XGB_OOF = [UCI / f"oof_xgb_{s}_{v}.npy" for s in ("enrol", "sem1", "multiclass")
           for v in ("fees", "nofee")]


UCI_TABLE_INPUTS = [UCI / f for f in ("cv_summary.csv", "cv_tests.csv", "analysis.json",
                                      "decision.json", "sensitivity.json", "transfer.json")]


@dataclass
class Step:
    name: str
    script: str
    about: str
    data: list[Path]
    needs: list[Path] = field(default_factory=list)  # files written by earlier steps


STEPS = [
    Step("uci.benchmark", "src/uci/01_benchmark.py",
         "8 models x 4 settings, 5x5 CV, corrected t-tests", UCI_FILES),
    Step("uci.calibration", "src/uci/02_calibration_subgroups_shap.py",
         "calibration, subgroups, feature selection, SHAP, figures", UCI_FILES,
         OOF + [UCI / "cv_summary.csv"]),
    Step("uci.fees", "src/uci/03_fee_sensitivity.py",
         "with and without the fee-status variables", UCI_FILES),
    Step("uci.alerts", "src/uci/04_alerts_and_thresholds.py",
         "fixed-size alert lists, group-specific thresholds", UCI_FILES, OOF + XGB_OOF),
    Step("uci.transfer", "src/uci/05_programme_transfer.py",
         "leave-one-programme-out validation", UCI_FILES),
    Step("uci.tuning", "src/uci/06_nested_tuning.py",
         "nested tuning: fixed vs tuned hyperparameters", UCI_FILES),
    Step("upv.validation", "src/upv/01_external_validation.py",
         "second university: CV, temporal, unseen degrees", UPV_FILES),
    Step("report.figures", "src/report/make_figures.py",
         "figures that compare the two universities", [], UCI_TABLE_INPUTS),
    Step("report.tables", "src/report/make_tables.py",
         "LaTeX tables and number macros (paper/generated/)", [], UCI_TABLE_INPUTS),
]

DATA_HELP = {
    "uci": "data/dataset.csv is part of the repository; restore it with:  git checkout data/dataset.csv",
    "upv": "download the three CSV files from https://doi.org/10.5281/zenodo.17239943 "
           "into data/upv/ (see data/README.md)",
}


# ── selection ───────────────────────────────────────────────────────────────
def resolve(names: list[str]) -> list[Step]:
    """Step names (uci.alerts) or groups (uci, upv, report)."""
    out = []
    for n in names:
        hits = [s for s in STEPS if s.name == n or s.name.startswith(n + ".")]
        if not hits:
            sys.exit(f"Unknown step '{n}'. See:  python run_all.py --list")
        out += hits
    return out


def select(args) -> list[Step]:
    chosen = {s.name for s in (resolve(args.only) if args.only else STEPS)}
    if args.from_step:
        start = STEPS.index(resolve([args.from_step])[0])
        chosen &= {s.name for s in STEPS[start:]}
    if args.skip:
        chosen -= {s.name for s in resolve(args.skip)}
    return [s for s in STEPS if s.name in chosen]


def missing(files: list[Path]) -> list[Path]:
    return [f for f in files if not f.exists()]


def rel(files: list[Path]) -> str:
    return ", ".join(str(f.relative_to(ROOT)) for f in files)


# ── reproducibility record ──────────────────────────────────────────────────
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*cmd: str) -> str:
    try:
        return subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment() -> dict:
    names = [line.split(">=")[0].split("==")[0].strip()
             for line in (ROOT / "requirements.txt").read_text().splitlines() if line.strip()]
    versions = {}
    for n in names:
        try:
            versions[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            versions[n] = "not installed"
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "commit": git("rev-parse", "--short", "HEAD"),
            "code_changed_since_commit": bool(git("status", "--porcelain", "--", "src", "run_all.py"))}


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


# ── running ─────────────────────────────────────────────────────────────────
def fmt(sec: float) -> str:
    return f"{sec / 60:.1f} min" if sec >= 60 else f"{sec:.0f} s"


def run(step: Step) -> float:
    t0 = time.time()
    with open(LOGS / f"{step.name}.log", "w") as log:
        # library deprecation notices from parallel workers would bury the progress lines
        env = os.environ | {"PYTHONWARNINGS": "ignore::FutureWarning,ignore::DeprecationWarning"}
        proc = subprocess.Popen([sys.executable, "-u", str(ROOT / step.script)], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write("  " + line)
            log.write(line)
        proc.wait()
    if proc.returncode:
        raise RuntimeError(step.name)
    return time.time() - t0


def build_paper() -> None:
    """Compile paper/main.tex in place (overwrites paper/main.pdf)."""
    if shutil.which("latexmk"):
        cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-quiet", "main.tex"]
    elif shutil.which("tectonic"):   # downloads the LaTeX packages it needs on first use
        cmd = ["tectonic", "--keep-logs", "main.tex"]
    else:
        print("No LaTeX found. Install one with  brew install tectonic  "
              "(or open paper/main.tex in a LaTeX editor such as Overleaf).")
        return
    print(f"\n── paper ({cmd[0]})")
    subprocess.run(cmd, cwd=ROOT / "paper", check=True)
    print("✓ paper/main.pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show the steps and exit")
    ap.add_argument("--only", nargs="+", metavar="STEP", help="run only these steps or groups")
    ap.add_argument("--skip", nargs="+", metavar="STEP", help="leave these steps out")
    ap.add_argument("--from", dest="from_step", metavar="STEP", help="start at this step")
    ap.add_argument("--paper", action="store_true", help="build paper/main.pdf at the end")
    args = ap.parse_args()

    steps = select(args)
    durations = load_json(DURATIONS)

    if args.list:
        print(f"  {'step':16s} {'last run':>9s}   what it does")
        for s in STEPS:
            on = "▸" if s in steps else " "
            t = fmt(durations[s.name]) if s.name in durations else "-"
            print(f"{on} {s.name:16s} {t:>9s}   {s.about}")
        print("\n▸ = would run with these options")
        return

    # Data check. Missing UPV files skip the UPV step, unless it was asked for by name.
    for group, files in (("uci", UCI_FILES), ("upv", UPV_FILES)):
        lost = missing(files)
        hit = [s for s in steps if s.name.startswith(group + ".")]
        if lost and hit:
            msg = f"missing {rel(lost)}\n  {DATA_HELP[group]}"
            if group == "uci" or args.only:
                sys.exit("Cannot start: " + msg)
            print(f"Skipping {group.upper()}: {msg}\n")
            steps = [s for s in steps if s not in hit]
    if not steps:
        sys.exit("Nothing to run.")

    LOGS.mkdir(parents=True, exist_ok=True)
    record = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "environment": environment(),
              "data_sha256": {str(f.relative_to(ROOT)): sha256(f)
                              for f in UCI_FILES + UPV_FILES if f.exists()},
              "steps": []}

    # Only a run of every step describes the committed results; partial runs
    # (--only, --skip, --from, missing data) are logged next to the step logs.
    full = len(steps) == len(STEPS)
    target = MANIFEST if full else LOGS / "last_partial_run.json"

    def save_record():
        target.write_text(json.dumps(record, indent=1) + "\n")
        DURATIONS.write_text(json.dumps(durations, indent=1) + "\n")

    print("Plan: " + " → ".join(s.name for s in steps) + "\n")
    t_all = time.time()
    for i, step in enumerate(steps, 1):
        lost = missing(step.needs)
        if lost:
            sys.exit(f"{step.name} needs {rel(lost)}, written by earlier steps.\n"
                     f"  Run the full pipeline once:  python run_all.py")
        last = f" · last time {fmt(durations[step.name])}" if step.name in durations else ""
        print(f"── [{i}/{len(steps)}] {step.name}: {step.about}{last}")
        try:
            sec = run(step)
        except RuntimeError:
            record["steps"].append({"name": step.name, "status": "failed"})
            save_record()
            sys.exit(f"\n✗ {step.name} failed. Full output: results/logs/{step.name}.log\n"
                     f"  After fixing it, resume with:  python run_all.py --from {step.name}")
        durations[step.name] = round(sec, 1)
        record["steps"].append({"name": step.name, "status": "ok", "seconds": round(sec, 1)})
        save_record()
        print(f"✓ {step.name} · {fmt(sec)}\n")

    record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_record()
    print("─" * 60)
    for s in record["steps"]:
        print(f"  {s['name']:16s} {fmt(s['seconds']):>9s}")
    print(f"  {'total':16s} {fmt(time.time() - t_all):>9s}")
    print(f"Run record: {target.relative_to(ROOT)}")

    if args.paper:
        build_paper()


if __name__ == "__main__":
    main()
