"""calculateFunctions.py

This module contains code for creating data products
useful to the QA/QC process that are not served
through OOI/M2M.

"""

import gc
import xarray as xr
import numpy as np
from rca_data_tools.qaqc.utils import select_logger, get_calibration_dataset, broadcast_calibrations, open_store, load_profiles
from rca_data_tools.qaqc.advanced_qaqc import nitrate
from rca_data_tools.qaqc.qartod import loadQARTOD, loadTestValues

logger = select_logger()


def combine_qc_flags(base_da, tests):
    """
    Combine a list of (name, mask) QC tests into a single positional string flag DataArray.

    Each flag is 1 (pass) or 3 (fail). The result is a string like "113" where each
    character corresponds to a test in order. Uses integer arithmetic to avoid dask
    string-concatenation issues.

    Parameters
    ----------
    base_da : xr.DataArray
        Template DataArray (e.g. xr.full_like(ds.some_var, 1)); determines shape/coords.
    tests : list of (str, xr.DataArray)
        Ordered list of (test_name, boolean_mask) where True means the point fails.

    Returns
    -------
    xr.DataArray
        String DataArray with attrs['tests_executed'] set to comma-joined test names.
    """
    n = len(tests)
    combined = sum(
        base_da.where(~mask, 3) * (10 ** (n - 1 - i))
        for i, (_, mask) in enumerate(tests)
    )
    result = combined.astype(int).astype(str)
    result.attrs['tests_executed'] = ','.join(name for name, _ in tests)
    return result


### -------------------------------------------------------------------------------------
### ADCP functions
### -------------------------------------------------------------------------------------

# Can use these based on NDBC but recommend using the
# TRDI QAQC Model rev12-1
ADCP_QCThresholds = {
    'error_velocity': {
        'pass': 0.05,
        'fail': 0.20 },
    'correlation_magnitude': {
        'pass': 115,
        'fail': 63 },
    'vertical_velocity': {
        'pass': 0.30,
        'fail': 0.50 },
    'horizontal_speed': {
        'pass': 1.00,
        'fail': 2.50 },
    'percent_good': {
        'ADCPT': {
            'pass': 56,
            'fail': 45 },
        'ADCPS': {
            'pass': 48,
            'fail': 38 }
    }
}

def sidelobe_depth(ds: xr.Dataset, theta: int = 20) -> xr.DataArray:
    """
    Calculate the sidelobe contamination depth for the given ADCP.

    The sidelobe interference depth is calculated following Lentz et al.
    (2022) where:

        z_ic = ha * [1 - cos(theta)] + 3 * delta_Z / 2

    z_ic is the depth above which there is sidelobe interference, ha is
    the transducer face depth, theta is the beam angle, and delta_Z is
    the cell-bin depth. Instrument tilt is ignored.

    Parameters
    ----------
    ds : xr.Dataset
        A TRDI ADCP dataset from OOI.
    theta : int, default 20
        Beam angle of the ADCP in degrees.

    Returns
    -------
    z_ic : xr.DataArray
        Sidelobe contamination depth (m).
    """
    ha = ds['depth_from_pressure'].chunk({'time': -1}).interpolate_na(dim='time', method='linear')
    theta = np.deg2rad(theta)
    delta_z = ds['cell_length'].mean(skipna=True) / 100   # cm → m
    z_ic = ha * (1 - np.cos(theta)) + 3 * delta_z / 2
    return z_ic

def adcp_advanced_flags(ds: xr.Dataset, instrument_type: str = 'ADCPT') -> xr.DataArray:
    """
    Assessment of TRDI ADCP data quality using QARTOD-style combined flags.

    Applies six tests — sidelobe contamination, error velocity, vertical
    velocity, horizontal speed, correlation magnitude, and percent good —
    and returns a positional string flag DataArray via combine_qc_flags.
    Thresholds are taken from ADCP_QCThresholds and follow the TRDI ADCP
    Data QA-QC Model rev12-1.

    Parameters
    ----------
    ds : xr.Dataset
        TRDI ADCP dataset downloaded from OOINet in NetCDF format.
    instrument_type : str, default 'ADCPT'
        Instrument subtype used to select percent-good thresholds;
        one of 'ADCPT' or 'ADCPS'.

    Returns
    -------
    xr.DataArray
        Positional string flag DataArray with attrs['tests_executed'].
    """
    thresholds = ADCP_QCThresholds

    # Sidelobe contamination — bins shallower than z_ic are flagged
    z_ic = sidelobe_depth(ds)
    sidelobe_mask = ds['bin_depths'] < z_ic

    # Error velocity
    ev_fail = thresholds['error_velocity']['fail']
    ev_mask = np.abs(ds['error_seawater_velocity']) > ev_fail

    # Vertical velocity
    vv_fail = thresholds['vertical_velocity']['fail']
    vv_mask = np.abs(ds['upward_seawater_velocity']) > vv_fail

    # Horizontal speed — fail if either east or north component exceeds threshold
    hs_fail = thresholds['horizontal_speed']['fail']
    hs_mask = (
        (np.abs(ds['eastward_seawater_velocity']) > hs_fail) |
        (np.abs(ds['northward_seawater_velocity']) > hs_fail)
    )

    # Correlation magnitude — fail if fewer than 2 of 4 beams pass
    cm_pass = thresholds['correlation_magnitude']['pass']
    beams = xr.concat([
        ds['correlation_magnitude_beam1'],
        ds['correlation_magnitude_beam2'],
        ds['correlation_magnitude_beam3'],
        ds['correlation_magnitude_beam4'],
    ], dim='beam')
    cm_mask = (beams > cm_pass).sum(dim='beam') < 2

# Percent good — fail if fewer than 3 of 4 beams pass
    pg_pass = thresholds['percent_good'][instrument_type]['pass']
    beams = xr.concat([
        ds['percent_good_beam1'],
        ds['percent_good_beam2'],
        ds['percent_good_beam3'],
        ds['percent_good_beam4'],
    ], dim='beam')
    pg_mask = (beams > pg_pass).sum(dim='beam') < 3

    base = xr.full_like(ds['eastward_seawater_velocity'], 1)
    tests = [
        ('sidelobe',             sidelobe_mask),
        ('error_velocity',       ev_mask),
        ('vertical_velocity',    vv_mask),
        ('horizontal_speed',     hs_mask),
        ('correlation_magnitude', cm_mask),
        ('percent_good',         pg_mask),
    ]
    return combine_qc_flags(base, tests)


### -------------------------------------------------------------------------------------
### FLOR functions
### -------------------------------------------------------------------------------------

def flor_advanced_flags(flor):
    """
    Assessment of the raw data and the calculated parameters for quality.
    Works for data streams with and without CDOM; CDOM tests are added 
    only when CDOM variables are present in the dataset.
    """
    max_counts = 4125   # counts should be greater than 0 and less than 4120 +/- 5

    # test raw signal ranges
    raw_backscatter_mask = (
        (flor.raw_signal_beta <= 0) | (flor.raw_signal_beta > max_counts)
    )
    raw_chlorophyll_mask = (
        (flor.raw_signal_chl <= 0) | (flor.raw_signal_chl > max_counts)
    )

    base = xr.full_like(flor.fluorometric_chlorophyll_a, 1)
    tests = [
        ('raw_signal_beta', raw_backscatter_mask),
        ('raw_signal_chl', raw_chlorophyll_mask),
    ]

    # CDOM raw signal test — only for data streams that include CDOM variables
    if 'raw_signal_cdom' in flor.variables:
        raw_cdom_mask = (
            (flor.raw_signal_cdom <= 0) | (flor.raw_signal_cdom > max_counts)
        )
        tests += [('raw_signal_cdom', raw_cdom_mask)]

    return combine_qc_flags(base, tests)


### -------------------------------------------------------------------------------------
### NUTNR functions
### -------------------------------------------------------------------------------------

def _profiler_ctd(site):
    """Reference designator of the profiler CTD on site's node."""
    from rca_data_tools.qaqc.constants import ALL_CONFIGS_DICT  # constants imports this module
    node = "-".join(site.split("-")[:2])
    return next(r for r, c in ALL_CONFIGS_DICT.items() if r.startswith(node) and c["instrument"] == "CTD-PROFILER")


def nutnr_advanced_flags(nutnr, site):
    """
    Per-sample SUNA spectral, instrument and CTD-input tests (1 pass, 3 flagged), ordered as
    attrs['tests_executed']. Missing nitrate is left to the QARTOD tests (flag 9). References:
        [CDOM note] Sea-Bird Scientific (2017). CDOM Interference on Nitrate Measurements
            (quoted elsewhere as SUNA180725).
        [Argo] Johnson et al. (2025). BGC-Argo quality control manual for nitrate
            concentration, v1.1. doi:10.13155/84370
        [SUNA] Sea-Bird Scientific (2025). SUNA V2 user manual, rev K. Its self-test reports
            the dark frame's average and SD across channels ("Spec Dark av sd", OOI SUNA driver).
        [EMVA] EMVA (2021). EMVA 1288 Standard for Characterization of Image Sensors and
            Cameras, Release 4.0 General, Section 2.3.
    """
    cals = get_calibration_dataset(site)
    dark = open_store(f"{site}-{nitrate.DARK_STREAM}")
    tests = [
        # A254/A350 > 1.3 AU: too little light reaches the detector [CDOM note]
        ('CDOM', (nutnr.nutnr_absorbance_at_254_nm > 1.3) | (nutnr.nutnr_absorbance_at_350_nm > 1.3)),
        # fit RMSE should be < 1E-3; higher means unmodelled absorbers [CDOM note; Argo 2.2.2.4]
        ('RMSE', nutnr.nutnr_fit_rmse > 0.001),  # onboard fit
        # absorbance near 240 nm > 0.8: fouled window or compromised optical path [Argo 2.2.2.3]
        ('A240', nitrate.absorbance_240(nutnr, cals) > 0.8),
        # any fit-window pixel above 64,500 counts: absorbance in error [Argo 2.2.2.2]
        ('saturation', nitrate.saturated_pixels(nutnr, cals) > 0),
        # raw spectrum identical to the adjacent sample's: stuck sensor output [Argo test 13]
        ('stuck', nitrate.stuck_spectrum(nutnr)),
        # the lamp should not operate above 35 °C [SUNA 8.3.2]
        ('lamp_temperature', nutnr.temp_lamp > 35),
        # replace the lamp after 750 hours [SUNA 8.1.2]
        ('lamp_hours', nutnr.lamp_time > 750 * 3600),
        # internal humidity > 50%: the sensor may have a leak [SUNA troubleshooting]
        ('humidity', nutnr.humidity > 50),
        # high dark counts (> 1000) fail the self-test [SUNA troubleshooting]
        ('dark', nutnr.nutnr_dark_value_used_for_fit > 1000),
        # 20x integration time in optically dense water [SUNA 8.4.1]: reprocessing, and the
        # recompute, assume the normal time [SUNA 6.2.3]
        ('integration_time', nutnr.nutnr_integration_time_factor > 1),
        # temporal noise of the sample's dark frame > 12 counts: its SD across channels exceeds
        # the self-test spec (<= 12) from noise alone [SUNA troubleshooting; EMVA Eqs. 17-18]
        ('dark_noise', nitrate.dark_noise(nutnr, dark) > 12),
    ]
    # the stream's T, S and P failing QARTOD gross range (the node's profiler CTD tables) flag
    # nitrate: 4 for T or P, 3 for S [Argo 2.2.2.1]; RCA flags 3 for all
    ctd = _profiler_ctd(site)
    for name, var, param in [('CTD_temperature', 'sea_water_temperature', 'seawater_temperature'),
                             ('CTD_salinity', 'sea_water_practical_salinity', 'practical_salinity'),
                             ('CTD_pressure', 'int_ctd_pressure', 'seawater_pressure')]:
        gross = loadTestValues(ctd, param, "ctdpf", "gross_range", logger=logger)
        tests.append((name, nitrate.gross_range_flag(nutnr[var], gross) == 4))
    if "nutnr_rmse" in nutnr:
        tests.append(('RMSE_recomputed', nutnr.nutnr_rmse > 0.001))  # nutnr_plant2023 fit

    base = xr.ones_like(nutnr.time, dtype=int)  # not NaN-masked, so missing nitrate can't corrupt the string
    return combine_qc_flags(base, tests)

def nutnr_plant2023(nutnr, site):
    """
    Dissolved nitrate [uM] recomputed from raw spectra: TCSS (Sakamoto et al. 2009) with the
    Plant et al. (2023) temperature correction and the Sakamoto et al. (2017) pressure
    correction. See nitrate.recompute_nitrate.

    Returns dissolved_nitrate, nutnr_fitting_function (nitrate, baseline intercept, baseline
    slope per nm) and nutnr_rmse; NaN for dark frames and times without a calibration.
    """
    out = nitrate.recompute_nitrate(nutnr, get_calibration_dataset(site))
    fit = xr.concat([out.nitrate, out.baseline_intercept, out.baseline_slope], dim="coefficient")
    return out.nitrate, fit.transpose("time", "coefficient"), out.fit_rmse


def nutnr_qartod_recomputed(nutnr, site):
    """
    OOI production QARTOD climatology, gross range and flat line tables (built for
    salinity_corrected_nitrate) applied to the recomputed dissolved_nitrate, then the profile
    tests (_nutnr_profile_tests). Returns OOI-style dissolved_nitrate_qartod_executed and
    dissolved_nitrate_qartod_results.
    """
    clim, gross = loadQARTOD(site, "salinity_corrected_nitrate", "nutnr", logger=logger)
    flat = loadTestValues(site, "salinity_corrected_nitrate", "nutnr", "flat_line", logger=logger)
    no3 = nutnr.dissolved_nitrate
    # every table is required, so a failed download raises rather than shifting the digits
    tests = {
        "climatology_test": nitrate.climatology_flag(no3, nutnr.int_ctd_pressure, clim),
        "gross_range_test": nitrate.gross_range_flag(no3, gross),
        "flat_line_test": nitrate.flat_line_flag(no3, flat),
    }
    tests.update(_nutnr_profile_tests(nutnr, site, gross))
    return nitrate.qartod_outputs(tests)


# per-sample tests a sample must pass to feed the profile-test statistics (the history
# the limits were derived from); instrument diagnostics such as dark_noise still flag it
SPECTRAL_TESTS = ["CDOM", "RMSE", "A240", "saturation", "stuck", "RMSE_recomputed"]


def _passes(flags, tests):
    """True where each of tests that advanced_qc_flags ran is 1."""
    names = flags.attrs["tests_executed"].split(",")
    return xr.concat([flags.str.get(names.index(t)) == "1" for t in tests if t in names], "test").all("test")


def _nutnr_profile_tests(nutnr, site, gross):
    """
    Spike (BGC-Argo test 9), multi-variate (IOOS test 9) and attenuated signal (IOOS test 10)
    flags for nutnr's samples, with limits from params/nutnr_profile_tests.yaml. Upcasts come
    from the RCA profile indices. Statistics use samples that pass the spectral tests
    (SPECTRAL_TESTS) and gross range. The previous history_days are read back from the store
    so the multi-variate test has its window.
    """
    from rca_data_tools.qaqc.constants import ALL_CONFIGS_DICT, NUTNR_PROFILE_TESTS as cfg  # constants imports this module
    keep = ["dissolved_nitrate", "sea_water_temperature", "int_ctd_pressure"]
    t0 = nutnr.time.values[0]
    start = t0 - np.timedelta64(cfg["multi_variate"]["history_days"], "D")
    stored = open_store(ALL_CONFIGS_DICT[site]["zarrFile"])
    parts = [nutnr[keep].assign(good=_passes(nutnr.advanced_qc_flags, SPECTRAL_TESTS))]
    if all(v in stored for v in keep + ["advanced_qc_flags"]):  # once harvests have stored the recomputed products
        old = stored[keep + ["advanced_qc_flags"]].sel(time=slice(start, t0 - np.timedelta64(1, "ns"))).load()
        parts.insert(0, old[keep].assign(good=_passes(old.advanced_qc_flags, SPECTRAL_TESTS)))
    d = xr.concat(parts, "time").load()

    no3, p, good = d.dissolved_nitrate, d.int_ctd_pressure, d.good
    good &= nitrate.gross_range_flag(no3, gross) != 4
    years = range(int(d.time.dt.year.min()) - 1, int(d.time.dt.year.max()) + 1)  # an upcast can start the year before
    table = nitrate.profile_table(load_profiles(site, years))
    upcast = nitrate.upcast_number(d.time, table)
    flags = {
        "spike_test": nitrate.spike_flag(no3, upcast, good, cfg["spike_uM"]),
        "multi_variate_test": nitrate.multi_variate_flag(no3, d.sea_water_temperature, p, upcast, good, table,
                                                         cfg["multi_variate"]),
        "attenuated_signal_test": nitrate.attenuated_signal_flag(no3, p, upcast, good, cfg["attenuated_signal"]["suspect_uM"][site],
                                                                 cfg["attenuated_signal"]["fail_uM"], cfg["full_depth"]),
    }
    return {name: f.sel(time=nutnr.time) for name, f in flags.items()}


### -------------------------------------------------------------------------------------
### OPTAA functions
### -------------------------------------------------------------------------------------

def opt_internal_temp(traw):
    """
    Description:

        Calculates the internal instrument temperature. Used in subsequent
        OPTAA calculations.  Copied from original code in
        https://bitbucket.org/ooicgsn/pyseas/get/develop.zip

    Implemented by:

        2013-04-25: Christopher Wingard. Initial implementation.
        2014-03-07: Russell Desiderio. Reduced calls to np.log.

    Usage:

        tintrn = opt_internal_temp(traw)

            where

        tintrn = calculated internal instrument temperature [deg_C]
        traw = raw internal instrument temperature (OPTTEMP_L0) [counts]

    References:

        OOI (2013). Data Product Specification for Optical Beam Attenuation
            Coefficient. Document Control Number 1341-00690.
            https://alfresco.oceanobservatories.org/ (See: Company Home >> OOI
            >> Controlled >> 1000 System Level >>
            1341-00690_Data_Product_SPEC_OPTATTN_OOI.pdf)

        OOI (2013). Data Product Specification for Optical Absorption
            Coefficient. Document Control Number 1341-00700.
            https://alfresco.oceanobservatories.org/ (See: Company Home >> OOI
            >> Controlled >> 1000 System Level >>
            1341-00700_Data_Product_SPEC_OPTABSN_OOI.pdf)
    """
    # convert counts to volts
    volts = 5.0 * traw / 65535.0

    # calculate the resistance of the thermistor
    res = 10000.0 * volts / (4.516 - volts)

    # convert resistance to temperature
    a = 0.00093135
    b = 0.000221631
    c = 0.000000125741

    log_res = np.log(res)
    degC = (1.0 / (a + b * log_res + c * log_res**3)) - 273.15
    
    ###logger.info(f"Calculated internal temperature: {degC.values}")
    
    return degC

def opt_external_temp(traw):
    """
    Description:

        Calculates the external environmental temperature of the ACS, if the unit
        is equipped with an auxiliary temperature sensor.  Copied from original code in
        https://bitbucket.org/ooicgsn/pyseas/get/develop.zip


    Implemented by:

        2013-04-25: Christopher Wingard. Initial implementation.
        2017-06-24: Christopher Wingard. Adjust for 32-bit execution

    Usage:

        textrn = opt_external_temp(traw)

            where

        textrn = calculated external environment temperature [deg_C]
        traw = raw external temperature [counts]

    References:

        OOI (2013). Data Product Specification for Optical Beam Attenuation
            Coefficient. Document Control Number 1341-00690.
            https://alfresco.oceanobservatories.org/ (See: Company Home >> OOI
            >> Controlled >> 1000 System Level >>
            1341-00690_Data_Product_SPEC_OPTATTN_OOI.pdf)

        OOI (2013). Data Product Specification for Optical Absorption
            Coefficient. Document Control Number 1341-00700.
            https://alfresco.oceanobservatories.org/ (See: Company Home >> OOI
            >> Controlled >> 1000 System Level >>
            1341-00700_Data_Product_SPEC_OPTABSN_OOI.pdf)
    """
    # convert counts to degrees Centigrade
    a = -7.1023317e-13
    b = 7.09341920e-08
    c = -3.87065673e-03
    d = 95.8241397

    # float64, not the original object dtype ("32-bit execution" workaround):
    # counts are 16-bit, so counts**3 < 2**53 is exact and the result is the same
    # bit for bit, without a python object per sample (~1.7x faster, less memory)
    t = traw.astype("float64")
    degC = a * t**3 + b * t**2 + c * traw + d
    
    ###logger.info(f"Calculated external temperature: {degC.values}")
    
    return degC.astype(float)

def opt_pressure(praw, offset, sfactor):
    """
    Description:

        Calculates the pressure (depth) of the ACS, if the unit is equipped
        with an auxiliary pressure sensor.  Copied from original code in
        https://bitbucket.org/ooicgsn/pyseas/get/develop.zip

    Implemented by:

        2013-04-25: Christopher Wingard. Initial implementation.

    Usage:

        depth = opt_pressure(praw, offset, sfactor)

            where

        depth = depth of the instrument [m]
        praw = raw pressure reading [counts]
        offset = depth offset from instrument device file [m]
        sfactor = scale factor from instrument device file [m counts-1]

    References:

        OOI (2013). Data Product Specification for Optical Beam Attenuation
            Coefficient. Document Control Number 1341-00690.
            https://alfresco.oceanobservatories.org/ (See: Company Home >> OOI
            >> Controlled >> 1000 System Level >>
            1341-00690_Data_Product_SPEC_OPTATTN_OOI.pdf)

        OOI (2013). Data Product Specification for Optical Absorption
            Coefficient. Document Control Number 1341-00700.
            https://alfresco.oceanobservatories.org/ (See: Company Home >> OOI
            >> Controlled >> 1000 System Level >>
            1341-00700_Data_Product_SPEC_OPTABSN_OOI.pdf)
    """
    depth = praw * sfactor + offset
    
    ###logger.info(f"Calculated depth: {depth.values}")
    
    return depth

def opt_calculate_all_optical_products(
    optaa,
    site,
    chl_line_height=0.020,
    time_chunk=1000000
):
    """
    Ultra memory-safe optical calculations.
    Uses calibration index lookup (no broadcast).
    Aggressively frees memory after each chunk.
    
    Pigment ratios can be calculated to assess the impacts of bio-fouling,
    sensor calibration drift, potential changes in community composition,
    light history or bloom health and age. Calculated ratios are:

    * CDOM Ratio -- ratio of CDOM absorption in the violet portion of the
        spectrum at 412 nm relative to chlorophyll absorption at 440 nm.
        Ratios greater than 1 indicate a preponderance of CDOM absorption
        relative to chlorophyll.
    * Carotenoid Ratio -- ratio of carotenoid absorption in the blue-green
        portion of the spectrum at 490 nm relative to chlorophyll absorption at
        440 nm. A changing carotenoid to chlorophyll ratio may indicate a shift
        in phytoplankton community composition in addition to changes in light
        history or bloom health and age.
    * Phycobilin Ratio -- ratio of phycobilin absorption in the green portion
        of the spectrum at 530 nm relative to chlorophyll absorption at 440 nm.
        Different phytoplankton, notably cyanobacteria, utilize phycobilins as
        accessory light harvesting pigments. An increasing phycobilin to
        chlorophyll ratio may indicate a shift in phytoplankton community
        composition.
    * Q Band Ratio -- the Soret and the Q bands represent the two main
        absorption bands of chlorophyll. The former covers absorption in the
        blue region of the spectrum, while the latter covers absorption in the
        red region. A decrease in the ratio of the intensity of the Soret band
        at 440 nm to that of the Q band at 676 nm may indicate a change in
        phytoplankton community structure. All phytoplankton contain
        chlorophyll 'a' as the primary light harvesting pigment, but green
        algae and dinoflagellates contain chlorophyll 'b' and 'c', respectively,
        which are spectrally redshifted compared to chlorophyll 'a'.      
    * Derive estimates of Chlorophyll-a and particulate organic carbon (POC)
        from scatter-corrected absorption and attenuation data.

    """
    logger.info("Loading calibrations")
    cals = get_calibration_dataset(site)

    logger.info("Building calibration index (no broadcast)...")
    coeffs = broadcast_calibrations(cals, optaa.time)
    cal_index = coeffs["index"]
    cal_tables = coeffs["tables"]

    logger.info("Finding wavelength indices...")
    # channel wavelengths of the calibration covering most samples; each instrument has its
    # own (NaN-padded) set, and the stored wavelength_a is a single vector for the whole record
    row = np.bincount(cal_index[cal_index >= 0]).argmax()
    aw_vals = cal_tables["CC_awlngth"][row]
    cw_vals = cal_tables["CC_cwlngth"][row]

    def nearest(vals, t): return int(np.nanargmin(np.abs(vals - t)))

    m412 = nearest(aw_vals, 412.0)
    m440 = nearest(aw_vals, 440.0)
    m490 = nearest(aw_vals, 490.0)
    m530 = nearest(aw_vals, 530.0)
    m676r = nearest(aw_vals, 676.0)

    m650 = nearest(aw_vals, 650.0)
    m676 = nearest(aw_vals, 676.0)
    m715 = nearest(aw_vals, 715.0)
    m660 = nearest(cw_vals, 660.0)

    apg = optaa["optical_absorption"]
    cpg = optaa["beam_attenuation"]

    time_dim = apg.dims[0]
    wl_dim_a = apg.dims[1]
    wl_dim_c = cpg.dims[1]

    n_time = optaa.sizes[time_dim]
    logger.info(f"Total time steps: {n_time}")

    # --- Preallocate outputs ---
    ratio_cdom  = np.full(n_time, np.nan, dtype="float32")
    ratio_carot = np.full(n_time, np.nan, dtype="float32")
    ratio_phyco = np.full(n_time, np.nan, dtype="float32")
    ratio_qband = np.full(n_time, np.nan, dtype="float32")
    chl         = np.full(n_time, np.nan, dtype="float32")
    poc         = np.full(n_time, np.nan, dtype="float32")

    def win3(i, n):
        return list(range(max(i-1, 0), min(i+2, n)))

    n_wl_a = apg.sizes[wl_dim_a]
    n_wl_c = cpg.sizes[wl_dim_c]

    idx650 = win3(m650, n_wl_a)
    idx676 = win3(m676, n_wl_a)
    idx715 = win3(m715, n_wl_a)
    idx660 = win3(m660, n_wl_c)

    # Build minimal wavelength request (deduplicated)
    wl_req = sorted(set(
        [m412, m440, m490, m530, m676r] +
        idx650 + idx676 + idx715
    ))

    pos = {w:i for i,w in enumerate(wl_req)}

    for start in range(0, n_time, time_chunk):
        stop = min(start + time_chunk, n_time)
        sl = slice(start, stop)

        #logger.info(f"Chunk {start}:{stop}")

        # ---- load minimal data ----
        abs_chunk = (
            apg.isel({time_dim: sl, wl_dim_a: wl_req})
               .data
               .compute()
        )
        att_chunk = (
            cpg.isel({time_dim: sl, wl_dim_c: idx660})
               .data
               .compute()
        )

        # ------------------------------------------------
        # Ratios
        # ------------------------------------------------
        a412 = abs_chunk[:, pos[m412]]
        a440 = abs_chunk[:, pos[m440]]
        a490 = abs_chunk[:, pos[m490]]
        a530 = abs_chunk[:, pos[m530]]
        a676v = abs_chunk[:, pos[m676r]]

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_cdom[sl]  = np.where(a440!=0, a412/a440, np.nan)
            ratio_carot[sl] = np.where(a440!=0, a490/a440, np.nan)
            ratio_phyco[sl] = np.where(a440!=0, a530/a440, np.nan)
            ratio_qband[sl] = np.where(a440!=0, a676v/a440, np.nan)

        # ------------------------------------------------
        # Chlorophyll (line height)
        # ------------------------------------------------
        a650 = np.nanmedian(abs_chunk[:, [pos[i] for i in idx650]], axis=1)
        a676 = np.nanmedian(abs_chunk[:, [pos[i] for i in idx676]], axis=1)
        a715 = np.nanmedian(abs_chunk[:, [pos[i] for i in idx715]], axis=1)

        abl = ((a715 - a650)/(715-650))*(676-650) + a650
        chl[sl] = (a676 - abl)/chl_line_height

        # ------------------------------------------------
        # POC
        # ------------------------------------------------
        poc[sl] = np.nanmedian(att_chunk, axis=1) * 381

        # ---- Aggressive cleanup ----
        del abs_chunk, att_chunk
        gc.collect()

    coords = {time_dim: optaa[time_dim]}

    logger.info("Wrapping outputs into DataArrays")

    # same order as the outputs in calculateCalls.csv
    return (
        xr.DataArray(chl,         coords=coords, dims=[time_dim], name="estimated_chlorophyll"),
        xr.DataArray(poc,         coords=coords, dims=[time_dim], name="estimated_poc"),
        xr.DataArray(ratio_cdom,  coords=coords, dims=[time_dim], name="ratio_cdom"),
        xr.DataArray(ratio_carot, coords=coords, dims=[time_dim], name="ratio_carotenoids"),
        xr.DataArray(ratio_phyco, coords=coords, dims=[time_dim], name="ratio_phycobilins"),
        xr.DataArray(ratio_qband, coords=coords, dims=[time_dim], name="ratio_qband"),
    )



### -------------------------------------------------------------------------------------
### pCO2 functions
### -------------------------------------------------------------------------------------

def pco2_test_function(ds, site):
    param_da = ds["pco2_seawater"]

    logger.info(f"generating a test array for site: {site}")
    test_da = xr.full_like(param_da, 1)

    test_da

    return test_da

def pco2w_advanced_flags(pco2w, site):
    """
    Assessment of the raw data and the calculated seawater pCO2 for quality.
    Test names encode severity: suspect_* for high-interest points, failed_* for
    outright failures. Each test is a binary pass/fail encoded as a positional
    character in the returned flag string (1 = pass, 3 = fail).

    :param pco2w: xarray dataset reformatted by pco2w_instrument or pco2w_datalogger
    :return: combined QC flag DataArray with attrs['tests_executed']
    """
    
    logger.info("Loading calibrations")
    cals = get_calibration_dataset(site)

    logger.info("Building calibration index (no broadcast)...")
    coeffs = broadcast_calibrations(cals, pco2w.time)
    cal_index = coeffs["index"]
    cal_tables = coeffs["tables"]

    calRange = cal_tables["CC_cal_range"]
    
     # unpack the light and reference measurements arrays into named variables
    light = pco2w.light_measurements.astype('int32')
    dark_reference = light[:, [0, 8]].values    # dark reference
    dark_signal = light[:, [1, 9]].values       # dark signal
    reference_434 = light[:, [2, 10]].values    # reference signal, 434 nm
    signal_434 = light[:, [3, 11]].values       # signal intensity, 434 nm
    reference_620 = light[:, [4, 12]].values    # reference signal, 620 nm
    signal_620 = light[:, [5, 13]].values       # signal intensity, 620 nm 
    
    # suspect dark reference & signal values -- values based on vendor documentation
    suspect_dark_mask = (
        (dark_reference < 50) | (dark_reference > 200)
    ).any(axis=1) | (
        (dark_signal < 50) | (dark_signal > 200)
    ).any(axis=1)

    # suspect signal levels -- values based on vendor documentation
    suspect_signal_mask = (
        (signal_434 > 4000) | (signal_620 > 4000)
    ).any(axis=1)

    # failed signal levels -- values based on limits used with the SAMI-pH data
    failed_signal_mask = (
        (signal_434 < 5) | (signal_620 < 5)
    ).any(axis=1)

    # failed absorbance blank ratio values (less than 20% of full scale)
    failed_blank_mask = (
        (pco2w.pco2w_a_absorbance_blank_434 < 16384 * 0.20) | (pco2w.pco2w_a_absorbance_blank_620 < 16384 * 0.20)
    )
    
    # abrupt steps in the absorbance blanks -- indicates solenoid pump failure to clear reagent/DI water
    failed_blank_step_mask = (
        (np.abs(pco2w.pco2w_a_absorbance_blank_434.diff('time')) > 2800) |
        (np.abs(pco2w.pco2w_a_absorbance_blank_620.diff('time')) > 2800)
    ).reindex_like(pco2w.pco2_seawater, fill_value=False)

    # abrupt steps in pCO2 -- indicates failure in raw parameters used in the ratio-based calculation
    failed_pco2_step_mask = (
        np.abs(pco2w.pco2_seawater.diff('time')) > 1600
    ).reindex_like(pco2w.pco2_seawater, fill_value=False)

    # pCO2 values outside the instrument calibration range
    cal_range_min = xr.DataArray(calRange[cal_index, 0], dims=["time"], coords={"time": pco2w.time})
    cal_range_max = xr.DataArray(calRange[cal_index, 1], dims=["time"], coords={"time": pco2w.time})
    failed_cal_range_mask = (
        (pco2w.pco2_seawater < cal_range_min) |
        (pco2w.pco2_seawater > cal_range_max)
    )

    base = xr.full_like(pco2w.pco2_seawater, 1)
    tests = [
        ('suspect_dark',       suspect_dark_mask),
        ('suspect_signal',     suspect_signal_mask),
        ('failed_signal',      failed_signal_mask),
        ('failed_blank',       failed_blank_mask),
        ('failed_blank_step',  failed_blank_step_mask),
        ('failed_pco2_step',   failed_pco2_step_mask),
        ('failed_cal_range',   failed_cal_range_mask),
    ]
    return combine_qc_flags(base, tests)



### -------------------------------------------------------------------------------------
### pH functions
### -------------------------------------------------------------------------------------

def ph_advanced_flags(ph):
    """
    adapted from https://github.com/oceanobservatories/ooi-data-explorations/blob/master/python/ooi_data_explorations/uncabled/process_phsen.py#L152
    """
    # unpack the light and reference measurements arrays into named variables
    nrec = len(ph['time'].values)
    light = np.array(np.vstack(ph['ph_light_measurements'].values), dtype='int32')
    light = np.atleast_3d(light)
    light = np.reshape(light, (nrec, 23, 4))  # 4 sets of 23 seawater measurements
    reference_434 = light[:, :, 0]            # reference signal, 434 nm
    signal_434 = light[:, :, 1]               # signal intensity, 434 nm (PH434SI_L0)
    reference_578 = light[:, :, 2]            # reference signal, 578 nm
    signal_578 = light[:, :, 3]               # signal intensity, 578 nm (PH578SI_L0)

    refrnc = np.array(np.vstack(ph['reference_light_measurements'].values), dtype='int32')
    refrnc = np.atleast_3d(refrnc)
    refrnc = np.reshape(refrnc, (nrec, 4, 4))   # 4 sets of 4 DI water measurements (blanks)
    blank_refrnc_434 = refrnc[:, :, 0]  # DI blank reference, 434 nm
    blank_signal_434 = refrnc[:, :, 1]  # DI blank signal, 434 nm
    blank_refrnc_578 = refrnc[:, :, 2]  # DI blank reference, 578 nm
    blank_signal_578 = refrnc[:, :, 3]  # DI blank signal, 578 nm

    # max measurement value
    max_bits = 4096

    # test suspect indicator signals -- values starting to get too low for a good calculation
    # values based on what would be considered too low for blanks, which should be the highest signals in the system.
    # If these are getting too low, it could indicate a problem with the pump or flow cell that is starting to
    # impact measurements, even if the blank measurements themselves haven't failed yet.

    low_indicator_signal_mask = ( (signal_434 < max_bits / 12).any(axis=1) |
                 (signal_578 < max_bits / 12).any(axis=1)
                )

    # test suspect flat indicator signals
    # indicates pump might be starting to fail or otherwise become obstructed.
    # if value is 3x the fail level, it is likely that the pump isn't completely failed yet,
    # but is starting to show signs of failure that could impact measurements.

    flat_indicator_signal_mask = ( (signal_434.std(axis=1) < 180) |
                 (signal_578.std(axis=1) < 180)
                )

    # test for suspect reference measurements
    # erratic reference measurements, with larger than usual variability
    # value based on 5x of normal standard deviations

    erratic_reference_mask = ( (reference_434.std(axis=1) > 10) |
                 (reference_578.std(axis=1) > 10)
                )

    # test for failed blank measurements -- blank measurements either too high (saturated signal) or too low.
    failed_blank_mask = ( (blank_signal_434 > max_bits - max_bits / 20).any(axis=1) |
                    (blank_signal_434 < max_bits / 12).any(axis=1) |
                    (blank_signal_578 > max_bits - max_bits / 20).any(axis=1) |
                    (blank_signal_578 < max_bits / 12).any(axis=1)
                )

    # test for failed intensity measurements -- intensity measurements either too high (saturated signal) or too low.
    failed_intensity_mask = ( (signal_434 > max_bits - max_bits / 20).any(axis=1) |
                    (signal_434 < 5).any(axis=1) |
                    (signal_578 > max_bits - max_bits / 20).any(axis=1) |
                    (signal_578 < 5).any(axis=1)
                )

    # test for flat intensity measurements -- indicates pump isn't working or the flow cell is otherwise obstructed.
    flat_intensity_mask = ( (signal_434.std(axis=1) < 60) |
                    (signal_578.std(axis=1) < 60)
                )

    base = xr.full_like(ph.ph_seawater, 1)
    tests = [
        ('low_indicator_signal',  low_indicator_signal_mask),
        ('flat_indicator_signal', flat_indicator_signal_mask),
        ('erratic_reference',     erratic_reference_mask),
        ('failed_blank',          failed_blank_mask),
        ('failed_intensity',      failed_intensity_mask),
        ('flat_intensity',        flat_intensity_mask),
    ]
    return combine_qc_flags(base, tests)



### -------------------------------------------------------------------------------------
### velpt functions
### -------------------------------------------------------------------------------------

def velpt_advanced_flags(velpt):
    """
    Assessment of pitch, roll, speed of sound, and pressure for the VELPT for quality.
    """
    # test for pitch out of range
    suspect_pitch_mask = np.abs(velpt.pitch_decidegree) > 20
    failed_pitch_mask = np.abs(velpt.pitch_decidegree) >= 30

    # test for roll out of range
    suspect_roll_mask = np.abs(velpt.roll_decidegree) > 20
    failed_roll_mask = np.abs(velpt.roll_decidegree) >= 30

    # test for valid speed of sound values (between 1400 and 1700 m/s)
    failed_speed_of_sound_mask = (velpt.sound_speed_dms <= 1400) | (velpt.sound_speed_dms >= 1700)

    # test for pressure out of range
    failed_pressure_mask = velpt.int_ctd_pressure <= 0
    
    base = xr.full_like(velpt.pitch_decidegree, 1)
    tests = [
        ('suspect_pitch',          suspect_pitch_mask),
        ('failed_pitch',           failed_pitch_mask),
        ('suspect_roll',           suspect_roll_mask),
        ('failed_roll',            failed_roll_mask),
        ('failed_speed_of_sound',  failed_speed_of_sound_mask),
        ('failed_pressure',        failed_pressure_mask),
    ]
    return combine_qc_flags(base, tests)



### -------------------------------------------------------------------------------------
### vel3d functions
### -------------------------------------------------------------------------------------

def vel3d_advanced_flags(vel3d):
    """
    Assessment of the pitch and roll and pressure values for the VEL3D.
    """
    base = xr.DataArray(
        np.ones(vel3d.time.size, dtype=int),
        dims=['time'],
        coords={'time': vel3d.time}
    )
    tests = []

    # test for pitch and roll out of range (greater than 30 degrees)
    if 'pitch' in vel3d.variables:
        tests += [
            ('suspect_pitch', np.abs(vel3d.pitch) > 20),
            ('failed_pitch',  np.abs(vel3d.pitch) >= 30),
        ]

    if 'roll' in vel3d.variables:
        tests += [
            ('suspect_roll', np.abs(vel3d['roll']) > 20),
            ('failed_roll',  np.abs(vel3d['roll']) >= 30),
        ]

    # test for valid speed of sound values (between 1400 and 1700 m/s)
    if 'speed_of_sound' in vel3d.variables:
        tests.append(('failed_speed_of_sound',
                      (vel3d.speed_of_sound < 1400) | (vel3d.speed_of_sound > 1700)))

    # test for pressure out of range
    if 'sea_water_pressure' in vel3d.variables:
        tests.append(('failed_pressure', vel3d.sea_water_pressure <= 15))

    # test for any error codes, which indicate a problem with the instrument
    if 'error_code' in vel3d.variables:
        tests.append(('failed_error_code', (vel3d.error_code.astype(int) & 1) == 1))

    # vector: test for low correlation values (less than 50%) from any of the three beams
    if 'correlation_beam1' in vel3d.variables:
        tests.append(('failed_correlation',
                      (vel3d.correlation_beam1 < 50) | (vel3d.correlation_beam2 < 50) | (vel3d.correlation_beam3 < 50)))

    # aquadopp: test for low correlation values (less than 50%) from any of the three data sets
    if 'correlation_1' in vel3d.variables:
        tests.append(('failed_correlation',
                      (vel3d.correlation_1 < 50) | (vel3d.correlation_2 < 50) | (vel3d.correlation_3 < 50)))

    # aquadopp: test for velocity values greater than the ambiguity velocity
    if 'ambiguity_velocity' in vel3d.variables:
        tests.append(('failed_ambiguity',
                      (np.abs(vel3d.velocity_1) > vel3d.ambiguity_velocity) |
                      (np.abs(vel3d.velocity_2) > vel3d.ambiguity_velocity) |
                      (np.abs(vel3d.velocity_3) > vel3d.ambiguity_velocity)))

    return combine_qc_flags(base, tests)