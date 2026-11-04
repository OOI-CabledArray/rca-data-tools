"""Config integrity: the params/ CSVs and YAMLs are the pipeline's source of
truth, so check that they reference each other consistently before a run does."""
import pandas as pd
import pytest

from rca_data_tools.qaqc.utils import ecs_job_variables
from rca_data_tools.qaqc.constants import (
    CALCULATE_CALLS_DICT,
    COMPUTE_EXCEPTIONS,
    FUNCTION_REGISTRY,
    HARVEST_COMPUTE,
    HARVEST_STREAMS,
    LOCAL_RANGE_DICT,
    PARAMS_DIR,
    SPAN_DICT,
)

csv = lambda name: pd.read_csv(PARAMS_DIR / name)
strip = lambda s: [v for v in str(s).strip('"').split(",") if v]

SITES = csv("sitesDictionary.csv")
ARCHIVE = csv("archiveDictionary.csv")
PLOT_PARAMS = csv("plotParameters.csv").set_index("instrument")["plotParameters"].map(strip)
VARIABLE_MAP = csv("variableMap.csv").set_index("parameter")["variableNames"].map(strip)
VARIABLE_PARAMS = csv("variableParameters.csv").set_index("variable")
PLOTTED = SITES[SITES.stage.isin([1, 2])]  # stage 3 cameras and stage 9 acoustics plot elsewhere
ALL_REFDES = set(SITES.refDes) | set(ARCHIVE.refDes)
ALL_ZARR = set(SITES.zarrFile) | set(ARCHIVE.zarrFile)


@pytest.mark.parametrize("name,df", [("sitesDictionary", SITES), ("archiveDictionary", ARCHIVE)])
def test_refdes_unique_and_complete(name, df):
    assert not df.refDes.duplicated().any(), f"{name} duplicate refDes: {sorted(df.refDes[df.refDes.duplicated()])}"
    plotted = df[df.stage.isin([1, 2])] if "stage" in df else df
    for col in ["instrument", "zarrFile", "dataParameters", "depths", "decimationAlgo"]:
        missing = plotted.refDes[plotted[col].isna()]
        assert missing.empty, f"{name} rows missing {col}: {sorted(missing)}"


def test_zarr_file_matches_refdes():
    # cameras (stage 3) and acoustics (stage 9) have no zarr store
    bad = [r.refDes for r in pd.concat([PLOTTED, ARCHIVE]).itertuples() if not str(r.zarrFile).startswith(r.refDes)]
    assert not bad, f"zarrFile does not start with its refDes: {bad}"


def test_stages_and_decimation_known():
    assert set(SITES.stage) <= {1, 2, 3, 9}, f"unknown stage: {set(SITES.stage) - {1, 2, 3, 9}}"
    algos = set(PLOTTED.decimationAlgo)
    assert algos <= {"lttb", "coarsen"}, f"unknown decimationAlgo: {algos - {'lttb', 'coarsen'}}"


def test_every_plotted_instrument_has_plot_parameters():
    missing = sorted(set(PLOTTED.instrument) - set(PLOT_PARAMS.index))
    assert not missing, f"instruments without plotParameters.csv rows: {missing}"


def test_every_plot_parameter_is_mapped():
    params = {p for ps in PLOT_PARAMS[PLOT_PARAMS.index.isin(PLOTTED.instrument)] for p in ps}
    assert not params - set(VARIABLE_MAP.index), f"not in variableMap.csv: {sorted(params - set(VARIABLE_MAP.index))}"
    assert not params - set(VARIABLE_PARAMS.index), f"not in variableParameters.csv: {sorted(params - set(VARIABLE_PARAMS.index))}"


# Fargate memory per vCPU count: (min GB, max GB, step GB)
FARGATE = {2: (4, 16, 1), 4: (8, 30, 1), 8: (16, 60, 4), 16: (32, 120, 8)}


def valid_fargate(size):
    job = ecs_job_variables(size)
    lo, hi, step = FARGATE.get(job["cpu"] // 1024, (1, 0, 1))
    gb = job["memory"] // 1024
    return lo <= gb <= hi and (gb - lo) % step == 0


def test_compute_exceptions_point_at_real_sites_spans_and_sizes():
    for site, spans in COMPUTE_EXCEPTIONS.items():
        assert site in ALL_REFDES, f"compute_exceptions.yaml: unknown site {site}"  # archived sites included
        for span, size in spans.items():
            assert str(span) in SPAN_DICT, f"compute_exceptions.yaml: {site} unknown span {span!r}"
            assert valid_fargate(size), f"compute_exceptions.yaml: {site} span {span} -> not a Fargate size {size}"


def test_harvest_streams():
    assert len(HARVEST_STREAMS) == len(set(HARVEST_STREAMS)), "harvest.yaml: extra stream duplicates a zarrFile"
    for t in HARVEST_STREAMS:
        assert t[:27] in set(PLOTTED.refDes), f"harvest.yaml: {t} is not a stage 1/2 instrument's stream"
        assert len(t[28:].split("-")) == 2, f"harvest.yaml: {t} is not <refDes>-<method>-<stream>"
    for t, sizes in HARVEST_COMPUTE.items():
        assert t in HARVEST_STREAMS or t in ALL_ZARR, f"harvest.yaml compute: unknown stream {t}"
        for kind, size in sizes.items():
            assert kind in {"refresh", "append"} and valid_fargate(size), f"harvest.yaml compute: {t} {kind}: {size}"


def test_site_calculations_resolve():
    calcs = csv("siteCalculations.csv")
    for r in calcs.itertuples():
        assert r.zarrFile in ALL_ZARR, f"siteCalculations.csv: unknown zarrFile {r.zarrFile}"
        for calc in str(r.calculations).split("|"):
            assert calc in CALCULATE_CALLS_DICT, f"siteCalculations.csv: {r.zarrFile} -> unknown calculation {calc}"
    for name, meta in CALCULATE_CALLS_DICT.items():
        assert meta["function_key"] in FUNCTION_REGISTRY, f"calculateCalls.csv: {name} -> no function {meta['function_key']}"


def test_deployed_ranges():
    for r in csv("deployedRanges.csv").itertuples():
        assert r.refDes in set(SITES.refDes), f"deployedRanges.csv: unknown refDes {r.refDes}"
        assert r.lowerB <= r.upperB, f"deployedRanges.csv: {r.refDes} lowerB > upperB"


def test_local_ranges():
    for site, params in LOCAL_RANGE_DICT.items():
        assert site in ALL_REFDES, f"localRanges.yaml: unknown refDes {site}"
        for param, ranges in params.items():
            assert param in VARIABLE_MAP.index, f"localRanges.yaml: {site} unknown parameter {param}"
            for kind, (lo, hi) in ranges.items():
                assert kind in {"local", "local_profile"} and lo < hi, f"localRanges.yaml: {site} {param} {kind} {[lo, hi]}"


def test_event_presets():
    for r in csv("eventPresets.csv").itertuples():
        assert r.refDes in ALL_REFDES, f"eventPresets.csv: unknown refDes {r.refDes}"
