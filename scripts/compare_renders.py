"""Render one instrument with two versions of the code and byte-compare the plots.

The check for any change that touches plotting or data handling: a refactor
should give identical PNG/SVG output, an intended change should differ only in
the files you expect. Reads real data from s3 (needs AWS credentials), so it is
a dev tool, not part of CI.

    python scripts/compare_renders.py RS03AXPS-PC03A-4A-CTDPFA303 7
    python scripts/compare_renders.py RS03AXPS-SF03A-2A-CTDPFA302 1 --params temperature --base main

Compares the working tree against --base (default: main). Matplotlib's SVG ids
and dates are pinned and TZ is UTC (as on ECS), so identical code gives
identical bytes. Exit status 1 if anything differs.
"""
import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

RENDER = """
import sys, matplotlib
matplotlib.rcParams["svg.hashsalt"] = "compare"
from rca_data_tools.qaqc.constants import ALL_CONFIGS_DICT, INSTRUMENT_DICT
from rca_data_tools.qaqc.plots import run_dashboard_creation
site, span, params, time_ref = sys.argv[1:5]
inst = ALL_CONFIGS_DICT[site]["instrument"]
params = params.split(",") if params else INSTRUMENT_DICT[inst]["plotParameters"].replace('"', "").split(",")
run_dashboard_creation(site, params, time_ref, inst, span, 5000000, ALL_CONFIGS_DICT, False, False)
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("site")
    ap.add_argument("span", choices=["0", "1", "7", "30", "365"])
    ap.add_argument("--params", default="", help="comma-separated plot parameters (default: all)")
    ap.add_argument("--time", default=time.strftime("%Y-%m-%dT00:00", time.gmtime()), help="timeRef, UTC")
    ap.add_argument("--base", default="main", help="git ref to compare the working tree against")
    ap.add_argument("--out", default="/tmp/compare_renders", help="scratch dir (envs are reused between runs)")
    a = ap.parse_args()

    out = Path(a.out)
    base_tree = out / "base-tree"
    if base_tree.exists():
        subprocess.run(["git", "-C", REPO, "worktree", "remove", "--force", base_tree], check=False)
    subprocess.run(["git", "-C", REPO, "worktree", "add", "--detach", base_tree, a.base], check=True)

    env = {**os.environ, "SOURCE_DATE_EPOCH": "0", "TZ": "UTC"}
    env.pop("VIRTUAL_ENV", None)
    jobs = {}
    try:
        for name, tree in [("base", base_tree), ("new", REPO)]:
            plots = out / name
            shutil.rmtree(plots, ignore_errors=True)
            plots.mkdir(parents=True)
            cmd = ["uv", "run", "--locked", "--project", str(tree), "python", "-c", RENDER,
                   a.site, a.span, a.params, a.time]
            log = open(plots / "run.log", "w")
            jobs[name] = (subprocess.Popen(cmd, cwd=plots, stdout=log, stderr=subprocess.STDOUT,
                                           env={**env, "UV_PROJECT_ENVIRONMENT": str(out / f"venv-{name}")}),
                          time.time(), plots)
        for name, (proc, t0, plots) in jobs.items():
            if proc.wait():
                sys.exit(f"{name} render failed, see {plots / 'run.log'}")
            print(f"{name}: rendered in {time.time() - t0:.0f}s")
    finally:
        subprocess.run(["git", "-C", REPO, "worktree", "remove", "--force", base_tree], check=False)

    base, new = out / "base" / "QAQC_plots", out / "new" / "QAQC_plots"
    names_b = {p.name for p in base.glob("*")} if base.exists() else set()
    names_n = {p.name for p in new.glob("*")} if new.exists() else set()
    differ = sorted(n for n in names_b & names_n if not filecmp.cmp(base / n, new / n, shallow=False))
    only_b, only_n = sorted(names_b - names_n), sorted(names_n - names_b)
    print(f"{len(names_b & names_n) - len(differ)}/{len(names_b | names_n)} files identical")
    for label, files in [("differ", differ), ("only in base", only_b), ("only in new", only_n)]:
        for f in files:
            print(f"  {label}: {f}")
    print(f"plots: {base}  vs  {new}")
    sys.exit(1 if differ or only_b or only_n else 0)


if __name__ == "__main__":
    main()
