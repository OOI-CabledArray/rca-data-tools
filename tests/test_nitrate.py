import io

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rca_data_tools.qaqc import qartod
from rca_data_tools.qaqc.advanced_qaqc import calculateFunctions, nitrate

WL = np.linspace(190, 394, 256)
NO3, TEMP, SAL, PRES, TCAL, DARK = 25.0, 8.0, 33.8, 150.0, 20.0, 600.0
INTERCEPT, SLOPE = 0.02, -5e-5


def make_cal():
    """One calibration row shaped like utils.get_calibration_dataset output."""
    eno3 = 0.03 * np.exp(-(WL - 200) / 8)
    eswa = 0.02 * np.exp(-(WL - 200) / 6)
    spectra = {"CC_wl": WL, "CC_eno3": eno3, "CC_eswa": eswa, "CC_di": np.full(256, 30000.0)}
    data = {k: (("calibration", f"{k}_index"), v[None]) for k, v in spectra.items()}
    data["CC_cal_temp"] = ("calibration", [TCAL])
    return xr.Dataset(data, coords={
        "calibration": [np.datetime64("2025-01-01")],
        "valid_start": ("calibration", [np.datetime64("2025-01-01")]),
        "valid_stop": ("calibration", [np.datetime64("2026-01-01")]),
    })


def make_suna(cal, temp_model="plant2023", pressure=True):
    """Spectra from known nitrate, baseline, T, S, P; the last sample is outside the calibration."""
    time = pd.date_range("2025-06-01", periods=4, freq="1min").append(pd.DatetimeIndex(["2026-06-01"]))
    c = cal.isel(calibration=0)
    x = WL - 210
    if temp_model == "plant2023":
        esw = c.CC_eswa.values * np.exp(sum(k * x**i for i, k in enumerate(nitrate.PLANT2023)) * (TEMP - TCAL))
    else:
        s = nitrate.SAKAMOTO2009
        esw = c.CC_eswa.values * (s["A"] + s["B"] * TEMP) / (s["A"] + s["B"] * TCAL) * np.exp(s["D"] * (TEMP - TCAL) * x)
    if pressure:
        esw = esw * (1 - nitrate.K_PRESSURE * PRES / 1000)
    absorbance = NO3 * c.CC_eno3.values + SAL * np.where(WL <= 240, esw, 0) + INTERCEPT + SLOPE * WL
    spectrum = DARK + c.CC_di.values * 10 ** -absorbance
    n = time.size
    spectra = np.tile(spectrum, (n, 1))
    spectra[:, int(np.argmax(WL >= 300))] += np.arange(n)  # detector noise outside the fit window; no two spectra identical
    return xr.Dataset({
        "spectral_channels": (("time", "wavelength"), spectra),
        "nutnr_dark_value_used_for_fit": ("time", np.full(n, DARK)),
        "sea_water_temperature": ("time", np.full(n, TEMP)),
        "sea_water_practical_salinity": ("time", np.full(n, SAL)),
        "int_ctd_pressure": ("time", np.full(n, PRES)),
        "frame_type": ("time", ["SLF", "SLF", "SLF", "SDF", "SLF"]),
        "nutnr_spectrum_average": ("time", np.full(n, 20000.0)),
        "nutnr_absorbance_at_254_nm": ("time", np.full(n, 0.2)),
        "nutnr_absorbance_at_350_nm": ("time", np.full(n, 0.1)),
        "nutnr_fit_rmse": ("time", np.full(n, 2e-4)),
        "temp_lamp": ("time", np.full(n, 15.0)),
        "lamp_time": ("time", np.full(n, 1e6)),
        "humidity": ("time", np.full(n, 2.5)),
        "nutnr_integration_time_factor": ("time", np.full(n, 1.0)),
    }, coords={"time": time, "wavelength": np.arange(256)})


def make_dark(times, noise):
    """Dark-frame stream with a fixed pattern (SD ~28 across channels) that is the same in every
    frame; frame k differs from frame k-1 by temporal noise noise[k] (noise[0] is unused)."""
    pattern = np.tile([1.0, -1.0], 128)
    frames = DARK + 40 * np.sin(np.arange(256)) + np.cumsum(np.outer(np.sqrt(2) * np.asarray(noise, float), pattern), axis=0)
    return xr.Dataset({"spectral_channels": (("time", "wavelength"), frames)},
                      coords={"time": pd.DatetimeIndex(times), "wavelength": np.arange(256)})


@pytest.mark.parametrize("temp_model,pressure", [("plant2023", True), ("sakamoto2009", False)])
def test_recompute_recovers_known_nitrate(temp_model, pressure):
    cal = make_cal()
    out = nitrate.recompute_nitrate(make_suna(cal, temp_model, pressure), cal, temp_model, pressure)
    np.testing.assert_allclose(out.nitrate[:3], NO3, rtol=1e-8)
    np.testing.assert_allclose(out.baseline_intercept[:3], INTERCEPT, rtol=1e-6)
    np.testing.assert_allclose(out.baseline_slope[:3], SLOPE, rtol=1e-6)
    assert float(out.fit_rmse[:3].max()) < 1e-10


def test_dark_frames_and_uncalibrated_times_are_nan():
    cal = make_cal()
    out = nitrate.recompute_nitrate(make_suna(cal), cal)
    assert out.nitrate[3].isnull() and out.nitrate[4].isnull()  # dark frame; outside calibration
    assert np.array_equal(nitrate.calibration_index(cal, out.time), [0, 0, 0, 0, -1])


def test_pressure_term_raises_nitrate():
    # ignoring pressure over-subtracts sea salt, biasing nitrate low
    cal = make_cal()
    ds = make_suna(cal, pressure=True)
    no_p = nitrate.recompute_nitrate(ds, cal, pressure=False).nitrate[0]
    assert float(no_p) < NO3


def test_saturation_counts_fit_window_only():
    cal = make_cal()
    ds = make_suna(cal)
    ds.spectral_channels[0, int(np.argmax(WL >= 300))] = 65000.0  # outside 217-240 nm: ignored
    ds.spectral_channels[1, int(np.argmax(WL >= 230))] = 65000.0
    counts = nitrate.saturated_pixels(ds, cal).values
    assert counts[:3].tolist() == [0, 1, 0] and np.isnan(counts[4])


def test_stuck_spectrum_flags_both_identical_neighbours():
    ds = make_suna(make_cal())
    ds["spectral_channels"][2] = ds.spectral_channels[1]
    ds["spectral_channels"][3, 0] = np.nan  # a NaN channel never counts as stuck
    ds["spectral_channels"][4] = ds.spectral_channels[3]
    assert nitrate.stuck_spectrum(ds).values.tolist() == [False, True, True, False, False]


def test_dark_noise_is_temporal_noise_of_each_samples_dark_frame():
    ds = make_suna(make_cal())
    t, s = ds.time.values, np.timedelta64(1, "s")
    # sample 1's frame pairs with sample 0's, 1 min earlier (sparse darks). Sample 2's frame is
    # older than DARK_AGE; sample 4's previous frame is a year older than DARK_PAIR: NaN
    times = [t[0] - 5 * s, t[0] - 3 * s, t[1] - 3 * s, t[2] - 20 * s, t[3] - 5 * s, t[3] - 3 * s, t[4] - 3 * s]
    dark = make_dark(times, [0, 5, 13, 0, 0, 5, 20])
    assert float(dark.spectral_channels[0].std()) > 12  # the fixed pattern cancels
    noise = nitrate.dark_noise(ds, dark).values
    np.testing.assert_allclose(noise[[0, 1, 3]], [5, 13, 5])
    assert np.isnan(noise[2]) and np.isnan(noise[4])


def test_absorbance_240_is_measured_seawater_absorbance():
    cal = make_cal()
    ds = make_suna(cal)
    ds["spectral_channels"][1] = DARK + (ds.spectral_channels[1] - DARK) * 0.1  # +1 AU, e.g. fouled window
    last = WL[WL <= 240][-1]
    c = cal.isel(calibration=0)
    plant = np.exp(sum(k * (last - 210) ** i for i, k in enumerate(nitrate.PLANT2023)) * (TEMP - TCAL))
    esw = np.interp(last, WL, c.CC_eswa.values) * plant * (1 - nitrate.K_PRESSURE * PRES / 1000)
    expected = NO3 * np.interp(last, WL, c.CC_eno3.values) + SAL * esw + INTERCEPT + SLOPE * last
    a240 = nitrate.absorbance_240(ds, cal).values
    np.testing.assert_allclose(a240[0], expected, rtol=1e-8)
    np.testing.assert_allclose(a240[1], expected + 1, rtol=1e-8)


GROSS = {"qartod": {"gross_range_test": {"suspect_span": [0.25, 36.4], "fail_span": [-1.5, 300]}}}
# as loaded by qartod.loadQARTOD: month -> pressure bin -> "[lo, hi]"
CLIM = {str(m): {"[0, 1]": "[0.0, 30.0]", "[1, 2]": "[nan, nan]"} for m in range(1, 13)}


def test_qartod_flags_on_recomputed_nitrate():
    time = pd.date_range("2025-06-01", periods=6, freq="1min")
    no3 = xr.DataArray([10.0, 33.0, 40.0, 400.0, np.nan, 10.0], coords={"time": time}, dims="time")
    pres = xr.DataArray([0.5, 0.5, 0.5, 0.5, 0.5, 1.5], coords={"time": time}, dims="time")
    tests = {"climatology_test": nitrate.climatology_flag(no3, pres, CLIM),
             "gross_range_test": nitrate.gross_range_flag(no3, GROSS)}
    assert tests["climatology_test"].values.tolist() == [1, 3, 3, 3, 9, 2]  # 1.5 dbar bin has no span
    assert tests["gross_range_test"].values.tolist() == [1, 1, 3, 4, 9, 1]
    executed, results = nitrate.qartod_outputs(tests)
    assert executed.attrs["tests_executed"] == "climatology_test, gross_range_test"
    assert executed.values.tolist() == ["11", "31", "33", "34", "99", "21"]
    assert results.values.tolist() == [1, 3, 3, 4, 9, 1]  # a pass outranks not evaluated (ioos_qc)


# 40 s samples: 120 s / 200 s are 3 / 5 previous points, the IOOS (2018) example counts
FLAT = {"qartod": {"flat_line_test": {"suspect_threshold": 120, "fail_threshold": 200, "tolerance": 1e-4}}}


def test_flat_line_flags_repeats_within_tolerance():
    time = pd.date_range("2025-06-01", periods=12, freq="40s")
    no3 = xr.DataArray([1, 2, 3, 3, 3, 3, 3.00005, 3, 4, np.nan, 5, 5], coords={"time": time}, dims="time")
    assert nitrate.flat_line_flag(no3, FLAT).values.tolist() == [1, 1, 1, 1, 1, 3, 3, 4, 1, 9, 1, 1]
    # a lone value after a gap passes (ioos_qc flags it 3)
    gap = xr.DataArray([1, np.nan, np.nan, np.nan, 2], coords={"time": time[:5]}, dims="time")
    assert nitrate.flat_line_flag(gap, FLAT).values.tolist() == [1, 9, 9, 9, 1]
    # sub-second sampling: 200 s is 400 previous points
    fast = xr.DataArray(np.r_[np.arange(100.0), np.full(500, 3.0)], dims="time",
                        coords={"time": pd.date_range("2025-06-01", periods=600, freq="500ms")})
    assert (nitrate.flat_line_flag(fast, FLAT) == 4).values.tolist() == [False] * 500 + [True] * 100


def test_flat_line_table_lookup(monkeypatch):
    csv = (
        "subsite,node,sensor,stream,parameters,qcConfig\n"
        "CE04OSPS,SF01B,4A-NUTNRA102,s,\"{'inp': 'nitrate_concentration'}\",\"{'qartod': {'flat_line_test': {'tolerance': 1}}}\"\n"
        "CE04OSPS,SF01B,4A-NUTNRA102,s,\"{'inp': 'salinity_corrected_nitrate'}\",\"{'qartod': {'flat_line_test': {'tolerance': 2}}}\"\n"
    )
    monkeypatch.setattr(qartod, "_load_table", lambda url: pd.read_csv(io.StringIO(csv)))
    got = qartod.loadTestValues("CE04OSPS-SF01B-4A-NUTNRA102", "salinity_corrected_nitrate", "nutnr", "flat_line")
    assert got["qartod"]["flat_line_test"]["tolerance"] == 2
    assert qartod.loadTestValues("RS01SBPS-SF01A-4A-NUTNRA101", "salinity_corrected_nitrate", "nutnr", "flat_line") == {}


# OOI production gross range for the profiler CTDs: fail span only
CTD_FAIL = {"seawater_temperature": [-5, 35], "practical_salinity": [0, 42], "seawater_pressure": [0, 6000]}


def test_advanced_flags_one_string_per_sample(monkeypatch):
    cal = make_cal()
    monkeypatch.setattr(calculateFunctions, "get_calibration_dataset", lambda site: cal)
    def ctd_table(refDes, param, sensorType, test, logger=None):
        assert (refDes, sensorType, test) == ("CE04OSPS-SF01B-2A-CTDPFA107", "ctdpf", "gross_range")
        return {"qartod": {"gross_range_test": {"fail_span": CTD_FAIL[param]}}}
    monkeypatch.setattr(calculateFunctions, "loadTestValues", ctd_table)
    ds = make_suna(cal)
    def dark_store(zarr_file):
        assert zarr_file == "CE04OSPS-SF01B-4A-NUTNRA102-streamed-nutnr_a_dark_sample"
        t = ds.time.values[:4]
        times = np.sort(np.concatenate([t - np.timedelta64(5, "s"), t - np.timedelta64(3, "s")]))
        return make_dark(times, [0, 5, 0, 5, 0, 5, 0, 13])
    monkeypatch.setattr(calculateFunctions, "open_store", dark_store)
    ds["nutnr_absorbance_at_254_nm"][0] = 1.5
    ds["nutnr_dark_value_used_for_fit"][0] = 1100.0
    ds["nutnr_dark_value_used_for_fit"][2] = 450.0  # low darks pass
    ds["sea_water_temperature"][0] = 40.0  # outside the CTD fail span
    ds["spectral_channels"][1] = DARK + (ds.spectral_channels[1] - DARK) * 0.1  # A240 > 0.8
    ds["nutnr_integration_time_factor"][1] = 20.0
    ds["sea_water_practical_salinity"][1] = 45.0
    ds.spectral_channels[2, int(np.argmax(WL >= 230))] = 65000.0  # saturated fit-window pixel
    ds["int_ctd_pressure"][2] = -1.0
    ds["temp_lamp"][3] = 36.0
    ds["lamp_time"][4] = 750 * 3600 + 1
    ds["humidity"][4] = 60.0
    flags = calculateFunctions.nutnr_advanced_flags(ds, "CE04OSPS-SF01B-4A-NUTNRA102")
    assert flags.dims == ("time",)
    assert flags.attrs["tests_executed"] == (
        "CDOM,RMSE,A240,saturation,stuck,lamp_temperature,lamp_hours,humidity,dark,integration_time,"
        "dark_noise,CTD_temperature,CTD_salinity,CTD_pressure")
    assert flags.values.tolist() == [
        "31111111311311", "11311111131131", "11131111111113", "11111311113111", "11111133111111"]


# profile tests: upcasts of 100 samples from 190 to 10 dbar, 39 s apart, 12 h apart
P = np.linspace(190, 10, 100)
FULL_DEPTH = {"top_dbar": 30, "bottom_dbar": 150, "min_samples": 40}


def make_upcasts(n, no3_offset=None, t_offset=None, gap_h=12, first=100):
    """Upcast samples and their RCA profile index table (utils.load_profiles)."""
    starts = pd.Timestamp("2025-06-01T07:00") + pd.to_timedelta(np.arange(n) * gap_h, "h")
    table = pd.DataFrame({"profile": first + np.arange(n), "start": starts,
                          "peak": starts + np.timedelta64(66, "m"), "end": starts + np.timedelta64(96, "m")})
    time = (starts.values[:, None] + (30 + 39 * np.arange(100)) * np.timedelta64(1, "s")).ravel()
    no3_off = np.zeros(n) if no3_offset is None else np.asarray(no3_offset, float)
    t_off = np.zeros(n) if t_offset is None else np.asarray(t_offset, float)
    ds = xr.Dataset({
        "dissolved_nitrate": ("time", (30 - 0.15 * (190 - P) + no3_off[:, None]).ravel()),
        "sea_water_temperature": ("time", (7 + (190 - P) / 30 + t_off[:, None]).ravel()),
        "int_ctd_pressure": ("time", np.tile(P, n)),
        "advanced_qc_flags": ("time", np.full(n * 100, "111111"), {"tests_executed": "CDOM,RMSE,A240,saturation,stuck,dark_noise"}),
    }, coords={"time": time})
    return ds, table


def test_upcast_number_is_start_to_peak():
    ds, table = make_upcasts(2)
    t = pd.DatetimeIndex(["2025-06-01T06:59", "2025-06-01T07:00", "2025-06-01T08:06", "2025-06-01T08:07", "2025-06-01T19:30"])
    time = xr.DataArray(t, coords={"time": t}, dims="time")
    up = nitrate.upcast_number(time, nitrate.profile_table(table))
    assert up.values.tolist() == [-1, 100, 100, -1, 101]  # before the first start, start, peak, after peak


def test_spike_flags_outliers_along_each_upcast():
    ds, table = make_upcasts(1)
    no3 = ds.dissolved_nitrate.copy()
    no3[50] += 6  # spike
    no3[20] += 4  # under 5 uM
    no3[40] = np.nan
    good = xr.ones_like(no3, dtype=bool)
    good[30] = False
    up = nitrate.upcast_number(ds.time, nitrate.profile_table(table))
    f = nitrate.spike_flag(no3, up, good, 5.0).values
    assert f[50] == 3 and f[20] == 1 and f[49] == 1 and f[51] == 1
    assert f[[0, 1, 98, 99]].tolist() == [2, 2, 2, 2]  # upcast ends
    assert f[30] == 2 and f[40] == 9  # not screened; missing
    assert set(f[2:98]) <= {1, 2, 3, 9} and (f == 3).sum() == 1


def test_attenuated_signal_flags_flat_upcasts():
    ds, table = make_upcasts(4)
    no3 = ds.dissolved_nitrate.copy()
    alt = np.tile([1.0, -1.0], 50)
    no3[100:200] = 20 + 0.1 * alt   # SD 0.1: fail
    no3[200:300] = 20 + 1.0 * alt   # SD 1.0: suspect
    good = (ds.int_ctd_pressure < 100) | (ds.time < ds.time[300])  # upcast 3 screened below 100 dbar only: not full depth
    up = nitrate.upcast_number(ds.time, nitrate.profile_table(table))
    f = nitrate.attenuated_signal_flag(no3, ds.int_ctd_pressure, up, ~good | (ds.time < ds.time[300]), 1.9, 0.3, FULL_DEPTH)
    per_upcast = [np.unique(f.values[i * 100:(i + 1) * 100]).tolist() for i in range(4)]
    assert per_upcast == [[1], [4], [3], [2]]


MV = {"n_nitrate": 4, "n_temperature": 1, "floor_uM": 0.3, "bin_dbar": 5, "pressure_dbar": [15, 185],
      "window_upcasts": 4, "min_upcasts": 3, "max_gap_hours": 14, "history_days": 8}


def test_multi_variate_flags_nitrate_steps_temperature_lacks():
    n = 8
    alt = 0.2 * (-1.0) ** np.arange(n)
    t_alt = 0.02 * (-1.0) ** np.arange(n)
    t_alt[-1] = t_alt[-2]  # last upcast: no temperature change
    ds, table = make_upcasts(n, no3_offset=alt, t_offset=t_alt)
    last = slice((n - 1) * 100, n * 100)
    p = ds.int_ctd_pressure
    step = (p >= 100) & (p < 105) & (ds.time >= ds.time[last.start])
    gated = (p >= 50) & (p < 55) & (ds.time >= ds.time[last.start])
    ds["dissolved_nitrate"] = ds.dissolved_nitrate + 5 * step + 5 * gated
    ds["sea_water_temperature"] = ds.sea_water_temperature + 1.0 * gated  # temperature moves with nitrate here
    good = xr.ones_like(p, dtype=bool)
    table_x = nitrate.profile_table(table)
    up = nitrate.upcast_number(ds.time, table_x)
    f = nitrate.multi_variate_flag(ds.dissolved_nitrate, ds.sea_water_temperature, p, up, good, table_x, MV)
    assert (f.where(step, drop=True) == 3).all() and (f.where(gated, drop=True) == 1).all()
    assert (f[last].where(~step & (p >= 15) & (p < 185), drop=True) == 1).all()
    assert (f[:300] == 2).all()  # no SD history yet
    assert (f[last].where((p < 15) | (p >= 185), drop=True) == 2).all()  # outside the bins
    # a gap longer than max_gap_hours: nothing to compare with
    ds2, table2 = make_upcasts(n, no3_offset=alt, t_offset=t_alt, gap_h=24)
    t2 = nitrate.profile_table(table2)
    f2 = nitrate.multi_variate_flag(ds2.dissolved_nitrate, ds2.sea_water_temperature, ds2.int_ctd_pressure,
                                    nitrate.upcast_number(ds2.time, t2), xr.ones_like(ds2.int_ctd_pressure, dtype=bool),
                                    t2, MV)
    assert (f2 == 2).all()
    # a partial upcast still has its bins compared
    shallow_gone = good & ~((p < 100) & (ds.time >= ds.time[last.start]))
    f3 = nitrate.multi_variate_flag(ds.dissolved_nitrate, ds.sea_water_temperature, p, up, shallow_gone, table_x, MV)
    assert (f3.where(step, drop=True) == 3).all()


def test_profile_tests_read_history_from_the_store(monkeypatch):
    n = 12
    t_alt = 0.02 * (-1.0) ** np.arange(n)
    t_alt[-1] = t_alt[-2]  # temperature varies between upcasts but not in the last one
    ds, table = make_upcasts(n, no3_offset=0.2 * (-1.0) ** np.arange(n), t_offset=t_alt)
    p = ds.int_ctd_pressure
    chunk_start = ds.time[(n - 1) * 100]
    ds["dissolved_nitrate"] = ds.dissolved_nitrate + 5 * ((p >= 100) & (p < 105) & (ds.time >= chunk_start))
    # dark_noise fails throughout: an instrument diagnostic, so samples still feed the statistics
    ds["advanced_qc_flags"] = ds.advanced_qc_flags.copy(data=np.full(n * 100, "111113"))
    history, chunk = ds.where(ds.time < chunk_start, drop=True), ds.where(ds.time >= chunk_start, drop=True)
    site = "CE04OSPS-SF01B-4A-NUTNRA102"
    def store(zarr_file, bucket="ooi-data"):
        assert (zarr_file, bucket) == (f"{site}-streamed-nutnr_a_sample", "rca-advanced-qaqc")
        return history
    monkeypatch.setattr(calculateFunctions, "open_store", store)
    monkeypatch.setattr(calculateFunctions, "load_profiles", lambda refDes, years: table)
    monkeypatch.setattr(calculateFunctions, "loadQARTOD", lambda *a, **k: (CLIM, GROSS))
    monkeypatch.setattr(calculateFunctions, "loadTestValues", lambda *a, **k: FLAT)
    executed, results = calculateFunctions.nutnr_qartod_recomputed(chunk, site)
    assert executed.attrs["tests_executed"] == ("climatology_test, gross_range_test, flat_line_test, spike_test, "
                                                "multi_variate_test, attenuated_signal_test")
    step = ((chunk.int_ctd_pressure >= 100) & (chunk.int_ctd_pressure < 105)).values
    mv = np.array([s[4] for s in executed.values])
    assert set(mv[step]) == {"3"} and set(mv[~step]) <= {"1", "2"}  # the window comes from the stored upcasts
    assert executed.sizes["time"] == chunk.sizes["time"]

    # a table that failed to download raises, so the digits can't shift
    monkeypatch.setattr(calculateFunctions, "loadTestValues", lambda *a, **k: {})
    with pytest.raises(KeyError):
        calculateFunctions.nutnr_qartod_recomputed(chunk, site)


def test_profile_tests_without_screened_upcast_samples_are_not_evaluated():
    ds, table = make_upcasts(2)
    late = ds.assign_coords(time=ds.time + np.timedelta64(30, "D"))  # newer than the profile indices
    late["dissolved_nitrate"][3] = np.nan
    t = nitrate.profile_table(table)
    up = nitrate.upcast_number(late.time, t)
    good = xr.ones_like(late.dissolved_nitrate, dtype=bool)
    expected = np.full(late.sizes["time"], 2)
    expected[3] = 9
    for f in (nitrate.spike_flag(late.dissolved_nitrate, up, good, 5.0),
              nitrate.multi_variate_flag(late.dissolved_nitrate, late.sea_water_temperature, late.int_ctd_pressure,
                                         up, good, t, MV),
              nitrate.attenuated_signal_flag(late.dissolved_nitrate, late.int_ctd_pressure, up, good, 1.9, 0.3, FULL_DEPTH)):
        assert f.values.tolist() == expected.tolist()


def test_load_profiles_reads_each_year(monkeypatch):
    from rca_data_tools.qaqc import utils
    csv = {2025: "profile,start,peak,end\n2,2025-02-01 07:00,2025-02-01 08:00,2025-02-01 09:00\n",
           2024: "profile,start,peak,end\n1,2024-12-31 07:00,2024-12-31 08:00,2024-12-31 09:00\n"}
    class Response:
        def __init__(self, url):
            year = int(url[-8:-4])
            self.status_code, self.text = (200, csv[year]) if year in csv else (404, "")
    monkeypatch.setattr(utils.requests, "get", lambda url, headers=None: Response(url))
    profiles = utils.load_profiles("CE04OSPS-SF01B-4A-NUTNRA102", years=[2023, 2025, 2024])
    assert profiles.profile.tolist() == [1, 2] and str(profiles.start.dtype).startswith("datetime64")
    assert utils.load_profiles("CE04OSPS", years=[2023]) == []
