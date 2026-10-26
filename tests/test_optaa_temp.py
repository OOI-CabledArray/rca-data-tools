import numpy as np
import pytest
import xarray as xr

from rca_data_tools.qaqc.advanced_qaqc.calculateFunctions import opt_external_temp


def reference_ext_temp(traw):
    """The original object-dtype formula (exact integer powers), as the oracle."""
    a, b, c, d = -7.1023317e-13, 7.09341920e-08, -3.87065673e-03, 95.8241397
    return (a * np.power(traw.astype("O"), 3) + b * np.power(traw.astype("O"), 2) + c * traw + d).astype(float)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint16, np.int32])
def test_ext_temp_matches_original_over_16bit_counts(dtype):
    # every 16-bit count, incl. the 65535 fill; stored data is float32 in 33609..65535
    counts = np.arange(0, 2**16).astype(dtype)
    got = np.asarray(opt_external_temp(xr.DataArray(counts, dims="time")))
    want = np.asarray(reference_ext_temp(xr.DataArray(counts, dims="time")))
    assert np.array_equal(got.view(np.uint64), want.view(np.uint64))


def test_ext_temp_nan_passthrough():
    traw = xr.DataArray(np.array([40000.0, np.nan], dtype=np.float32), dims="time")
    got = opt_external_temp(traw).values
    assert np.isnan(got[1]) and np.array_equal(got[:1], reference_ext_temp(traw).values[:1])
