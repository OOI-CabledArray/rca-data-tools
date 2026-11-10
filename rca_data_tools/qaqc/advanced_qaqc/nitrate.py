"""nitrate.py

Nitrate from raw SUNA spectra, per-sample spectral checks and profile tests for the RCA
shallow-profiler SUNAs (Stage 1 of the nitrate QAQC roadmap).

References:
    Johnson, K. S., and L. J. Coletti (2002). Deep-Sea Res. I 49:1291-1305.
    Sakamoto, C. M., K. S. Johnson, and L. J. Coletti (2009). Limnol. Oceanogr.: Methods 7:132-143.
    Sakamoto, C. M., et al. (2017). Limnol. Oceanogr.: Methods 15:897-902.
    Plant, J. N., et al. (2023). Limnol. Oceanogr.: Methods 21:581-593.
    U.S. IOOS (2018). Manual for Real-Time Quality Control of Dissolved Nutrients Observations, v1.1.
    Johnson, K. S., et al. (2025). BGC-Argo quality control manual for nitrate concentration, v1.1.
"""

import json

import numpy as np
import xarray as xr

from rca_data_tools.qaqc.utils import calibration_index

DARK_FRAMES = ["SDB", "SDF", "NDF"]
DARK_STREAM = "streamed-nutnr_a_dark_sample"  # RCA streams dark frames separately
DARK_AGE = np.timedelta64(10, "s")  # a sample's dark frame precedes it by 2-4 s
# longest gap to the previous dark frame: 2 s in bursts, 5-60 min in sparse setups. Pairs an
# hour apart give the same noise as consecutive frames (the fixed pattern holds)
DARK_PAIR = np.timedelta64(65, "m")
FIT_WINDOW = (217, 240)  # nm, first wavelength >= 217 to last <= 240 (Plant et al. 2023)
SATURATION = 64500  # counts (BGC-Argo nitrate QC manual, 2.2.2.2)

# Plant et al. 2023, Eq. 6: ESW(T) = ESW(Tcal) * exp(f(x) * (T - Tcal)), f(x) = sum(c_k * x**k), x = wl - 210
PLANT2023 = [1.46380e-02, 1.67660e-03, 2.91898e-05, -7.56395e-06, 1.27353e-07]
# Sakamoto et al. 2009, Eq. 4 (C cancels in F(T) / F(Tcal))
SAKAMOTO2009 = {"A": 1.1500276, "B": 0.02840, "C": -0.3101349, "D": 0.001222}
# Sakamoto et al. 2017: ESW(P) = ESW(1 dbar) * (1 - 0.026 * P / 1000), P in dbar
K_PRESSURE = 0.026


def by_calibration(ds, cals, func):
    """Apply func(ds_part, cal_row) to each calibration's samples; times without a
    calibration come back NaN."""
    idx = calibration_index(cals, ds.time)
    rows = np.unique(idx[idx >= 0]) if (idx >= 0).any() else [0]
    parts = [func(ds.isel(time=idx == i), cals.isel(calibration=i).reset_coords(drop=True)) for i in rows]
    return xr.concat(parts, "time").reindex(time=ds.time)


def _spectrum(cal, name):
    """1-D calibration array on the data's `wavelength` (channel) dim."""
    return cal[name].rename({f"{name}_index": "wavelength"})


def _window(cal, fit_window):
    wl = _spectrum(cal, "CC_wl")
    return ((wl >= fit_window[0]) & (wl <= fit_window[1])).values


def _dark_corrected(ds, win):
    return ds.spectral_channels.isel(wavelength=win) - ds.nutnr_dark_value_used_for_fit


def _tcss(ds, cal, temp_model, pressure, fit_window):
    """Temperature-compensated, salinity-subtracted fit for samples sharing one calibration."""
    win = _window(cal, fit_window)
    wl, eno3, eswa, di = (_spectrum(cal, n).isel(wavelength=win) for n in ("CC_wl", "CC_eno3", "CC_eswa", "CC_di"))
    intensity = _dark_corrected(ds, win)
    absorbance = -np.log10((intensity / di).where(intensity > 0))

    temp, tcal, x = ds.sea_water_temperature, cal.CC_cal_temp, wl - 210
    if temp_model == "plant2023":
        esw = eswa * np.exp(sum(c * x**k for k, c in enumerate(PLANT2023)) * (temp - tcal))
    elif temp_model == "sakamoto2009":
        a, b, d = SAKAMOTO2009["A"], SAKAMOTO2009["B"], SAKAMOTO2009["D"]
        esw = eswa * (a + b * temp) / (a + b * tcal) * np.exp(d * (temp - tcal) * x)
    else:
        raise ValueError(f"unknown temp_model {temp_model!r}")
    if pressure:
        esw = esw * (1 - K_PRESSURE * ds.int_ctd_pressure / 1000)
    esw = esw.where(wl <= 240, 0)  # no sea-salt extinction above 240 nm (Plant et al. 2023)
    acomp = absorbance - ds.sea_water_practical_salinity * esw

    # nitrate plus linear baseline; 1/100 and wl/1000 scale the columns as in OOI's L2 code
    design = xr.concat([eno3, xr.ones_like(eno3) / 100, wl / 1000], dim="coefficient")
    pinv = xr.DataArray(np.linalg.pinv(design.transpose("wavelength", "coefficient").values),
                        dims=("coefficient", "wavelength"))
    coef = xr.dot(pinv, acomp, dim="wavelength")
    residual = acomp - xr.dot(design, coef, dim="coefficient")
    return xr.Dataset({
        "nitrate": coef.isel(coefficient=0),
        "baseline_intercept": coef.isel(coefficient=1) / 100,
        "baseline_slope": coef.isel(coefficient=2) / 1000,
        "fit_rmse": np.sqrt((residual**2).mean("wavelength")),
    })


def recompute_nitrate(ds, cals, temp_model="plant2023", pressure=True, fit_window=FIT_WINDOW):
    """
    Nitrate [uM] from raw spectra: absorbance against the calibration reference, sea-salt
    extinction corrected for temperature (temp_model "plant2023" or "sakamoto2009") and
    pressure, subtracted, then nitrate and a linear baseline fit by least squares.

    Uses spectral_channels, nutnr_dark_value_used_for_fit, sea_water_temperature,
    sea_water_practical_salinity, int_ctd_pressure and frame_type from ds, and calibrations
    from utils.get_calibration_dataset. temp_model="sakamoto2009", pressure=False reproduces
    OOI L2 salinity_corrected_nitrate.

    Returns a Dataset of nitrate, baseline_intercept, baseline_slope and fit_rmse; NaN for
    dark frames and times without a calibration.
    """
    out = by_calibration(ds, cals, lambda d, c: _tcss(d, c, temp_model, pressure, fit_window))
    return out.where(~ds.frame_type.isin(DARK_FRAMES))


def saturated_pixels(ds, cals, fit_window=FIT_WINDOW):
    """
    Raw fit-window pixels above SATURATION; their absorbance is in error (BGC-Argo nitrate
    QC manual, 2.2.2.2).
    """
    def count(d, c):
        return (d.spectral_channels.isel(wavelength=_window(c, fit_window)) > SATURATION).sum("wavelength")
    return by_calibration(ds, cals, count)


def stuck_spectrum(ds):
    """
    Raw spectrum identical in every channel to the previous or next sample's (BGC-Argo
    nitrate QC manual, test 13). The first sample of a chunk can't see the previous chunk.
    """
    s = ds.spectral_channels
    return (s == s.shift(time=1)).all("wavelength") | (s == s.shift(time=-1)).all("wavelength")


def absorbance_240(ds, cals, fit_window=FIT_WINDOW):
    """
    Seawater absorbance at the last fit pixel <= 240 nm. Neither nitrate nor bromide absorbs
    strongly there, so high values mean a fouled window or too little light reaching the
    detector (BGC-Argo nitrate QC manual, 2.2.2.3).
    """
    def a240(d, c):
        last = np.flatnonzero(_window(c, fit_window))[-1:]
        intensity = _dark_corrected(d, last)
        a = -np.log10((intensity / _spectrum(c, "CC_di").isel(wavelength=last)).where(intensity > 0))
        return a.isel(wavelength=0, drop=True)
    return by_calibration(ds, cals, a240)


def dark_noise(ds, dark):
    """
    Temporal noise of the dark frame each sample was corrected with (the last one before it:
    its average is nutnr_dark_value_used_for_fit): SD across channels of its difference from
    the previous dark frame, / sqrt(2) (EMVA 1288 Eq. 18). The fixed pattern cancels, so this
    is at most the SD across channels the SUNA self-test reports (Eq. 17). dark is the
    dark-frame stream; NaN where the sample has no dark frame within DARK_AGE, or that frame
    no previous one within DARK_PAIR.
    """
    t = ds.time.values
    frames = dark.spectral_channels.sel(time=slice(t[0] - DARK_AGE - DARK_PAIR, t[-1]))
    noise = (frames.diff("time").std("wavelength") / np.sqrt(2)).where(frames.time.diff("time") <= DARK_PAIR)
    return noise.reindex(time=ds.time, method="ffill", tolerance=DARK_AGE)


def gross_range_flag(x, table):
    """QARTOD gross range from an OOI production table: 4 outside fail_span, 3 outside
    suspect_span (optional, as in ioos_qc; the profiler CTD tables have none)."""
    t = table["qartod"]["gross_range_test"]
    (fail_lo, fail_hi), (sus_lo, sus_hi) = t["fail_span"], t.get("suspect_span", t["fail_span"])
    flag = xr.where((x < fail_lo) | (x > fail_hi), 4, xr.where((x < sus_lo) | (x > sus_hi), 3, 1))
    return flag.where(x.notnull(), 9)


def flat_line_flag(x, table):
    """QARTOD flat line from an OOI production table (IOOS 2018, test 8), as ioos_qc runs it:
    thresholds in seconds become point counts via the median sample interval; 4 (3) where the
    values over the fail (suspect) window span less than the tolerance. The first window's
    points pass, so a chunk's first points can't see the previous chunk. Unlike ioos_qc, a
    window needs two valid values: a lone value after a gap has nothing to repeat."""
    t = table["qartod"]["flat_line_test"]
    interval = float(x.time.diff("time").median() / np.timedelta64(1, "s"))
    position = xr.ones_like(x.time, dtype=int).cumsum("time")
    def flat(seconds):
        n = int(seconds / interval)
        window = x.rolling(time=n + 1, min_periods=2)
        return ((window.max() - window.min()) < t["tolerance"]) & (position > n)
    flag = xr.where(flat(t["fail_threshold"]), 4, xr.where(flat(t["suspect_threshold"]), 3, 1))
    return flag.where(x.notnull(), 9)


def profile_table(profiles):
    """RCA profile indices (utils.load_profiles) as a Dataset on `upcast`, the profile number."""
    return profiles.set_index("profile").to_xarray().rename(profile="upcast")


def upcast_number(time, table):
    """Profile number of each sample on an upcast (start <= t <= peak in the profile
    indices), -1 elsewhere."""
    by_start = table.set_coords("start").swap_dims(upcast="start").reset_coords("upcast")
    last = by_start.reindex(start=time.values, method="ffill").rename(start="time").assign_coords(time=time.time)
    return last.upcast.where(time <= last.peak, -1).astype(int).rename("upcast")


def _not_evaluated(no3):
    """2 everywhere, 9 where missing."""
    return xr.full_like(no3, 2, dtype=int).where(no3.notnull(), 9)


def spike_flag(no3, upcast, good, threshold):
    """BGC-Argo test 9 along each upcast: 3 where a screened (good) value differs from the
    5-point running median of its upcast's screened values by more than threshold. 2 for
    samples not screened, off an upcast or within two samples of an upcast's ends; 9 missing."""
    x = no3.where(good & (upcast >= 0)).dropna("time")
    if not x.size:
        return _not_evaluated(no3)
    ref = x.groupby(upcast.sel(time=x.time)).map(lambda g: g.rolling(time=5, center=True).median())
    flag = xr.where(abs(x - ref) > threshold, 3, 1).where(ref.notnull(), 2)
    return flag.reindex_like(no3, fill_value=2).where(no3.notnull(), 9)


def _full_depth_upcasts(no3, pressure, upcast, good, full_depth):
    """Per upcast: SD of its screened nitrate, and whether its screened samples span the
    full_depth criteria (top_dbar, bottom_dbar, min_samples)."""
    g = xr.Dataset({"no3": no3, "p": pressure}).where(good).groupby(upcast)
    n, top, bottom = g.count().no3, g.min().p, g.max().p
    full = ((n.upcast >= 0) & (n >= full_depth["min_samples"]) & (top <= full_depth["top_dbar"])
            & (bottom >= full_depth["bottom_dbar"]))
    return g.std().no3, full


def attenuated_signal_flag(no3, pressure, upcast, good, suspect, fail, full_depth):
    """IOOS test 10 per upcast: on every sample of a full-depth upcast, 4 where the SD of its
    screened nitrate is below fail, 3 below suspect, else 1. 2 off an upcast or on upcasts
    that aren't full depth; 9 missing."""
    sd, full = _full_depth_upcasts(no3, pressure, upcast, good, full_depth)
    per = xr.where(sd < fail, 4, xr.where(sd < suspect, 3, 1)).where(full, 2)
    return per.sel(upcast=upcast).drop_vars("upcast").where(no3.notnull(), 9)


def multi_variate_flag(no3, temp, pressure, upcast, good, table, cfg):
    """
    IOOS test 9: a nitrate step that temperature doesn't share. In each pressure bin of
    consecutive upcasts (cfg bin_dbar, pressure_dbar, max_gap_hours), 3 where
    |dNO3| > max(n_nitrate x SD_NO3, floor_uM) and |dT| < n_temperature x SD_T, else 1. Bin
    values are medians of screened samples; SDs are over the bin's previous window_upcasts (at
    least min_upcasts). Every sample in the bin gets its flag; 2 where not evaluated
    (no consecutive upcast or SD history, off an upcast, outside the bins); 9 missing.
    """
    lo, hi = cfg["pressure_dbar"]
    pbin = (pressure // cfg["bin_dbar"]).where((pressure >= lo) & (pressure < hi), -1).astype(int)
    key = (upcast * 1000 + pbin).where(good & (pbin >= 0) & (upcast >= 0), -1).rename("key")
    med = xr.Dataset({"no3": no3, "t": temp}).groupby(key).median()
    med = med.isel(key=(med.key >= 0).values)
    if not med.key.size:
        return _not_evaluated(no3)
    med = med.assign_coords(upcast=med.key // 1000, pbin=med.key % 1000).set_index(key=["upcast", "pbin"]).unstack("key")

    start = table.start.sel(upcast=med.upcast)
    consec = (start.diff("upcast", label="upper") <= np.timedelta64(cfg["max_gap_hours"], "h")).reindex_like(start, fill_value=False)

    def change(x):
        step = abs(x.diff("upcast", label="upper")).reindex_like(x).where(consec)
        sd = x.rolling(upcast=cfg["window_upcasts"], min_periods=cfg["min_upcasts"]).std().shift(upcast=1)
        return step, sd

    dn, sdn = change(med.no3)
    dt, sdt = change(med.t)
    ok = dn.notnull() & sdn.notnull() & dt.notnull() & sdt.notnull()
    hit = (dn > np.maximum(cfg["n_nitrate"] * sdn, cfg["floor_uM"])) & (dt < cfg["n_temperature"] * sdt)
    flag = xr.where(ok, xr.where(hit, 3, 1), 2).reindex(upcast=np.unique(upcast), pbin=np.unique(pbin), fill_value=2)
    return flag.sel(upcast=upcast, pbin=pbin).drop_vars(["upcast", "pbin"]).where(no3.notnull(), 9)


def _span(cell):
    s = np.array(str(cell).strip("[]").split(","), dtype=float)
    return s if s.size == 2 else np.full(2, np.nan)


def climatology_flag(x, pressure, table):
    """
    QARTOD climatology from an OOI production month x pressure-bin table (as loaded by
    qartod.loadQARTOD): 3 outside the bin's span, 2 where the table has no span.
    """
    bins = sorted(table["1"], key=lambda b: json.loads(b)[0])
    edges = np.array([json.loads(b) for b in bins], dtype=float)  # (bin, [top, bottom]) dbar
    spans = np.array([[_span(table[str(m)][b]) for b in bins] for m in range(1, 13)])  # (month, bin, [lo, hi])

    def flag(v, month, p):
        b = np.clip(np.searchsorted(edges[:, 0], p, side="right") - 1, 0, None)
        lo, hi = spans[month - 1, b, 0], spans[month - 1, b, 1]
        f = np.where((v < lo) | (v > hi), 3, 1)
        no_span = np.isnan(lo) | np.isnan(hi) | np.isnan(p) | (p < edges[0, 0]) | (p >= edges[b, 1])
        return np.where(np.isnan(v), 9, np.where(no_span, 2, f))

    return xr.apply_ufunc(flag, x, x.time.dt.month, pressure, dask="parallelized", output_dtypes=[int])


def qartod_outputs(tests):
    """
    OOI-style QARTOD variables from {test_name: flag}: a per-test digit string with
    attrs['tests_executed'], and a summary by ioos_qc's qartod_compare precedence:
    4 > 3 > 1 > 2 (not evaluated) > 9 (missing). OOI's stores take the highest flag instead,
    which lets a test that couldn't run override a pass.
    """
    flags = list(tests.values())
    executed = sum(f * 10 ** (len(flags) - 1 - i) for i, f in enumerate(flags)).astype(f"<U{len(flags)}")
    executed.attrs["tests_executed"] = ", ".join(tests)
    rank = xr.concat([xr.where(f == 9, -1, xr.where(f == 2, 0, f)) for f in flags], "test").max("test")
    results = xr.where(rank == -1, 9, xr.where(rank == 0, 2, rank)).astype("uint8")
    return executed, results
