"""Reproduce every result in reports/ from data/raw (after the Bloomberg pulls).

    python scripts/run_all.py            # all specifications + result sheet + data recap  (~5 min)
    python scripts/run_all.py --synthetic # same pipeline on generated data, no Bloomberg needed
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
K5 = "--knots 0,3,7,15,30"

# tag -> run_history arguments. R25_k5 is the main specification.
SPECS = {
    "R25_k5": f"--recovery 0.25 {K5}",
    "R25_k5_fundbtf": f"--recovery 0.25 {K5} --funding --funding-src btf3m",
    "R25_k5_funding": f"--recovery 0.25 {K5} --funding --funding-src gc",
    "R40": "--recovery 0.40 --knots 0,3,7",  # benchmark from the notes: 3 buckets
    "R40_k5": f"--recovery 0.40 {K5}",
    "R10_k5": f"--recovery 0.10 {K5}",
    "Rfit_k5": f"--recovery fit {K5}",
    "R25_k5_bund": f"--recovery 0.25 {K5} --base bund",
}
MAIN, COMPARE = "R25_k5", "R25_k5_fundbtf"

NOTE = (r"\textbf{Real Bloomberg data} (pulled 24 Nov 2026): 117 fixed-coupon nominal OATs, 2010 to today, weekly fits. "
        r"Main specification: recovery 25\% and five hazard buckets (0-3, 3-7, 7-15, 15-30, 30Y+). "
        r"Recovery 40\% cannot price the deep-discount ultra-long OATs (0.5\% 2072 at 23, 0.75\% 2053 at 37). "
        r"\textbf{Funding:} French GC repo history is not entitled on the terminal used; STOXX GC Pooling (ECB basket) "
        r"sits within a few bp of €STR and changes nothing. Using 3M BTF bills minus €STR as a collateral-scarcity proxy "
        r"(comparison run R25\_k5\_fundbtf) cuts the share of QE dates where the fit wants $\lambda<0$ from 77\% to 40\%, "
        r"and lifts the average 0-3Y $\lambda$ under QE from $-12$bp to $+3$bp.")


def run(cmd: str, *extra: str):
    print(">>", cmd, *(e[:60] + "..." for e in extra), flush=True)
    subprocess.run([PY, *shlex.split(cmd), *extra], check=True, cwd=ROOT)  # no shell: LaTeX $...$ stays intact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    if args.synthetic:
        run("scripts/make_synthetic.py")
        run("scripts/run_history.py --raw data/synthetic --tag synth")
        run("scripts/run_history.py --raw data/synthetic --tag synth_funding --funding")
        run("scripts/make_report.py --tag synth --compare synth_funding --raw data/synthetic --synthetic "
            "--outdir reports/synthetic")
        return
    for tag, a in SPECS.items():
        run(f"scripts/run_history.py {a} --tag {tag}")
    extra = ",".join(t for t in SPECS if t not in (MAIN, COMPARE))
    run(f"scripts/make_report.py --tag {MAIN} --compare {COMPARE} --extra {extra} --outdir reports/results",
        "--note", NOTE)
    run("scripts/make_data_sheet.py")


if __name__ == "__main__":
    main()
