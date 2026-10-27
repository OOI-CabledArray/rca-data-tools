import numpy as np
import xarray as xr

from rca_data_tools.qaqc.dashboard import maskTimeGaps
from rca_data_tools.qaqc.decimate import coarsen_by_segment


def adcp_like(times):
    n = len(times)
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {"velocity": (("time", "bin"), rng.normal(size=(n, 4))),
         "bin_depths": (("time", "bin"), np.tile([10.0, 20.0, 30.0, 40.0], (n, 1)))},
        coords={"time": times},
    )


def regular(start, n, step_s=1):
    return np.datetime64(start) + np.arange(n) * np.timedelta64(step_s, "s")


def test_no_gap_matches_plain_coarsen():
    ds = adcp_like(regular("2026-01-01", 10_000))
    xr.testing.assert_identical(coarsen_by_segment(ds, 1_000), ds.coarsen(time=10, boundary="trim").mean())


def test_no_column_straddles_a_gap():
    before, after = regular("2026-01-01", 5_005), regular("2026-01-10", 5_005)  # 9-day gap
    ds = adcp_like(np.concatenate([before, after]))
    out = coarsen_by_segment(ds, 1_000)
    t = out.time.values
    gap0, gap1 = before[-1], after[0]
    assert not ((t > gap0) & (t < gap1)).any(), "a column was placed inside the gap"
    # every column is the mean of samples from one side only
    left = ds.sel(time=slice(None, gap0)).coarsen(time=11, boundary="trim").mean()
    right = ds.sel(time=slice(gap1, None)).coarsen(time=11, boundary="trim").mean()
    xr.testing.assert_identical(out, xr.concat([left, right], dim="time"))


def test_segment_shorter_than_window_dropped():
    ds = adcp_like(np.concatenate([regular("2026-01-01", 5_000), regular("2026-02-01", 3)]))
    out = coarsen_by_segment(ds, 1_000)
    assert out.time.values.max() < np.datetime64("2026-01-02")


def test_mask_returns_inputs_unchanged_without_gaps():
    ds = adcp_like(regular("2026-01-01", 100))
    x, y, z = ds.time, ds.bin_depths.T, ds.velocity.T
    assert all(out is inp for out, inp in zip(maskTimeGaps(x, y, z), (x, y, z)))


def test_mask_blanks_gap():
    t = np.concatenate([regular("2026-01-01", 50, 60), regular("2026-01-05", 50, 60)])  # 1-min columns, multi-day gap
    ds = adcp_like(t)
    x, y, z = maskTimeGaps(ds.time, ds.bin_depths.T, ds.velocity.T)
    assert x.size == t.size + 2 and z.shape[1] == t.size + 2
    i = np.flatnonzero(np.isnan(z[0]))
    assert list(i) == [50, 51]                                     # two NaN columns, between the two sides
    assert x[50] == t[49] + np.timedelta64(60, "s") and x[51] == t[50] - np.timedelta64(60, "s")
    assert np.isfinite(y).all()                                     # pcolormesh needs finite depths
    assert np.array_equal(z[:, :50], ds.velocity.T.values[:, :50])  # data untouched


def test_sidelobe_cutoff_workhorse():
    from rca_data_tools.qaqc.dashboard import sidelobeCutoff
    t = regular("2026-01-01", 3)
    ds = xr.Dataset({"bin_depths": (("time", "bin"), np.tile([190.0, 182.0, 174.0], (3, 1))),
                     "depth_from_pressure": ("time", [200.0, 200.0, np.nan]),     # gap in the pressure record
                     "cell_length": ("time", [800.0, 800.0, 800.0])},            # cm
                    coords={"time": t})
    cut = sidelobeCutoff(ds, "bin_depths", "RS01SBPS-PC01A-05-ADCPTD102").values
    expect = 200 * (1 - np.cos(np.deg2rad(20))) + 1.5 * 8                  # ~24 m
    assert np.allclose(cut[:2], expect)
    assert np.isfinite(cut[2])                                             # falls back to bin geometry, not NaN


def test_sidelobe_cutoff_vadcp_from_bin_geometry():
    from rca_data_tools.qaqc.dashboard import sidelobeCutoff
    ds = xr.Dataset({"vadcp_b_bin_depths": (("time", "bin"), [[189.7, 189.2, 188.7]])},
                    coords={"time": regular("2026-01-01", 1)})
    cut = float(sidelobeCutoff(ds, "vadcp_b_bin_depths", "RS03AXPS-PC03A-06-VADCPB301")[0])
    h, dz = 189.7 + 0.5, 0.5                                               # deepest bin + one cell; bin spacing
    assert np.isclose(cut, h * (1 - np.cos(np.deg2rad(25))) + 1.5 * dz)   # ~18.6 m, 25 deg Sentinel V beams
