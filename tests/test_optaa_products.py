import numpy as np
import xarray as xr

from rca_data_tools.qaqc.advanced_qaqc import calculateFunctions
from rca_data_tools.qaqc.calculate import run_calculations_for_site

ZARR = "CE04OSPS-SF01B-3B-OPTAAD105-streamed-optaa_sample"
WL = np.r_[np.linspace(400, 733, 85), np.full(7, np.nan)]  # NaN-padded, as in asset-management tables
PEAK, CP = 0.01, 0.3


def make_cal():
    t = np.array(["2025-01-01"], dtype="datetime64[ns]")
    return xr.Dataset(
        {k: (("calibration", f"{k}_index"), WL[None]) for k in ("CC_awlngth", "CC_cwlngth")},
        coords={"calibration": t, "valid_start": ("calibration", t),
                "valid_stop": ("calibration", np.array(["2026-01-01"], dtype="datetime64[ns]"))},
    )


def test_products_land_in_their_configured_outputs(monkeypatch):
    monkeypatch.setattr(calculateFunctions, "get_calibration_dataset", lambda site: make_cal())
    a = np.tile(0.5 * np.exp(-0.015 * (WL - 400)), (2, 1))
    a[1, np.abs(WL - 676) < 6] += PEAK  # chlorophyll peak on the second sample only
    time = np.array(["2025-06-01T00:00", "2025-06-01T00:01"], dtype="datetime64[ns]")
    ds = xr.Dataset({"optical_absorption": (("time", "wavelength"), a),
                     "beam_attenuation": (("time", "wavelength"), np.full_like(a, CP))},
                    coords={"time": time}).chunk()
    out, params = run_calculations_for_site(ZARR, ds, {ZARR: ["opt_calculate_all_optical_products"]})
    near = lambda w: int(np.nanargmin(np.abs(WL - w)))  # noqa: E731
    np.testing.assert_allclose(out.estimated_chlorophyll.diff("time"), PEAK / 0.020, rtol=1e-5)
    np.testing.assert_allclose(out.estimated_poc, CP * 381, rtol=1e-6)
    np.testing.assert_allclose(out.ratio_cdom[0], a[0, near(412)] / a[0, near(440)], rtol=1e-6)
    np.testing.assert_allclose(out.ratio_qband[0], a[0, near(676)] / a[0, near(440)], rtol=1e-6)
