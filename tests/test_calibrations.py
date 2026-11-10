import numpy as np
import xarray as xr

from rca_data_tools.qaqc.utils import broadcast_calibrations, calibration_index


def cals(starts, stops):
    d = lambda s: np.array(s, dtype="datetime64[ns]")  # noqa: E731
    return xr.Dataset(
        {"CC_cal_range": (("calibration", "CC_cal_range_index"), [[100.0, 1000.0], [200.0, 2000.0]][: len(starts)])},
        coords={"calibration": d(starts), "valid_start": ("calibration", d(starts)), "valid_stop": ("calibration", d(stops))},
    )


def test_uncovered_times_get_nan_not_last_calibration():
    c = cals(["2020-01-01", "2021-01-01"], ["2020-06-01", "2022-01-01"])
    t = np.array(["2019-12-31", "2020-02-01", "2020-07-01", "2021-03-01", "2022-02-01"], dtype="datetime64[ns]")
    out = broadcast_calibrations(c, t)
    assert out["index"].tolist() == [-1, 0, -1, 1, -1]
    low = out["tables"]["CC_cal_range"][out["index"], 0]
    assert np.isnan(low[[0, 2, 4]]).all() and low[[1, 3]].tolist() == [100.0, 200.0]


def test_overlapping_windows_use_latest_start():
    # pandas IntervalIndex.get_indexer raised on overlaps; never fall back to a wrong row
    c = cals(["2020-01-01", "2020-03-01"], ["2020-12-01", "2020-06-01"])
    t = np.array(["2020-02-01", "2020-04-01", "2020-07-01"], dtype="datetime64[ns]")
    assert calibration_index(c, t).tolist() == [0, 1, -1]
