"""data-vent installs rca-data-tools from main and imports these names; keep them working,
and keep them free of the plotting stack."""
import inspect
import json
import subprocess
import sys

from rca_data_tools.qaqc.calculate import run_calculations_for_site
from rca_data_tools.qaqc.constants import HARVEST_COMPUTE, HARVEST_STREAMS, MAX_COORD_SIZES, PARAMS_DIR, S3_BUCKET  # noqa: F401
from rca_data_tools.qaqc.utils import get_s3_kwargs, load_site_calculations  # noqa: F401

PLOTTING = {"matplotlib", "rca_data_tools.qaqc.dashboard", "rca_data_tools.qaqc.plots"}


def test_data_vent_names():
    load_site_calculations(PARAMS_DIR / "siteCalculations.csv", during_harvest=True)
    assert "calculate_dict" in inspect.signature(run_calculations_for_site).parameters


def test_data_vent_imports_skip_plotting():
    # fresh interpreter: other tests have already loaded matplotlib
    code = ("import json, sys, rca_data_tools.qaqc.calculate, rca_data_tools.qaqc.constants, "
            "rca_data_tools.qaqc.utils; print(json.dumps(list(sys.modules)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    loaded = PLOTTING & set(json.loads(out.splitlines()[-1]))
    assert not loaded, f"data-vent imports load the plotting stack: {sorted(loaded)}"
