"""calculateFunctions.py

This module contains code for creating data products
useful to the QA/QC process that are not served
through OOI/M2M.

"""

import ast
import gc
import xarray as xr
import numpy as np
from typing import Dict, Optional
from rca_data_tools.qaqc.utils import select_logger, get_calibration_dataset, broadcast_calibrations

logger = select_logger()

###
### OPTAA functions
###

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

    degC = a * np.power(traw.astype("O"), 3) + b * np.power(traw.astype("O"), 2) + c * traw + d
    
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
    wl_a_vals = optaa["wavelength_a"].values
    aw_vals = cal_tables["CC_awlngth"]
    cw_vals = cal_tables["CC_cwlngth"]

    def nearest(vals, t): return int(np.abs(vals - t).argmin())

    m412 = nearest(wl_a_vals, 412.0)
    m440 = nearest(wl_a_vals, 440.0)
    m490 = nearest(wl_a_vals, 490.0)
    m530 = nearest(wl_a_vals, 530.0)
    m676r = nearest(wl_a_vals, 676.0)

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

    return (
        xr.DataArray(ratio_cdom,  coords=coords, dims=[time_dim], name="ratio_cdom"),
        xr.DataArray(ratio_carot, coords=coords, dims=[time_dim], name="ratio_carotenoids"),
        xr.DataArray(ratio_phyco, coords=coords, dims=[time_dim], name="ratio_phycobilins"),
        xr.DataArray(ratio_qband, coords=coords, dims=[time_dim], name="ratio_qband"),
        xr.DataArray(chl,         coords=coords, dims=[time_dim], name="estimated_chlorophyll"),
        xr.DataArray(poc,         coords=coords, dims=[time_dim], name="estimated_poc"),
    )


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


###
### pCO2 functions
###

def pco2_test_function(ds, site):
    param_da = ds["pco2_seawater"]

    logger.info(f"generating a test array for site: {site}")
    test_da = xr.full_like(param_da, 1)

    test_da

    return test_da



###
### pH functions
###

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


###
### NUTNR functions
###

def nutnr_advanced_flags(nutnr):
    
    # "Absorption: The data output of the SUNA V2 is the absorption at 350 nm and 254 nm
    # (A350 and A254). These wavelengths are outside the nitrate absorption range and can be
    # used to make an estimate of the impact of CDOM. If absorption is high (>1.3 AU), the
    # SUNA will not be able to collect adequate light to make a measurement." SUNA V2 vendor
    # documentation (Sea-Bird Scientific Document# SUNA180725)
    
    # "RMSE: The root-mean-square error parameter from the SUNA V2 can be used to make
    # an estimate of how well the nitrate spectral fit is. This should usually be less than 1E-3. If
    # it is higher, there is spectral shape (likely due to CDOM) that adversely impacts the nitrate
    # estimate." SUNA V2 vendor documentation (Sea-Bird Scientific Document# SUNA180725)

    # Invalid: points where spectra - dark is negative, inf, or nan
    invalid_mask = (
        ((nutnr.spectral_channels - nutnr.nutnr_dark_value_used_for_fit) <= 0) |
        np.isinf(nutnr.spectral_channels) |
        nutnr.spectral_channels.isnull()
    )
    # Blocked absorption channel or failed lamp
    channel_mask = (
        (nutnr.nutnr_spectrum_average < 10000) 
    )
    # CDOM interference: where A254 or A350 > 1.3 AU
    cdom_mask = (
        (nutnr.nutnr_absorbance_at_254_nm > 1.3) |
        (nutnr.nutnr_absorbance_at_350_nm > 1.3)
    )
    # onboard-RMSE: if rmse > 0.001; also include plant2023 RMSE if present
    rmse_mask = (nutnr.nutnr_fit_rmse > 0.001)
    if "nutnr_rmse" in nutnr:
        rmse_mask = rmse_mask | (nutnr.nutnr_rmse > 0.001)
        
    base = xr.full_like(nutnr.salinity_corrected_nitrate, 1)
    tests = [
        ('channel', channel_mask),
        ('invalid', invalid_mask),
        ('CDOM',    cdom_mask),
        ('RMSE',    rmse_mask),
    ]
    return combine_qc_flags(base, tests)

def nutnr_plant2023(nutnr, site):
    """
    Description:

        The code below calculates the Dissolved Nitrate Concentration
        with the Plant et al (2023) updates to the 
        Sakamoto et. al. (2009) algorithm that uses the observed
        sample salinity and temperature to subtract the bromide component
        of the overall seawater UV absorption spectrum before solving for
        the nitrate concentration. Additionally, this adds the pressure
        correction from Sakamoto 2017.

        The output represents the OOI L2 Dissolved Nitrate Concentration,
        Temperature and Salinity Corrected (NITRTSC).

    Implemented by:

        2014-05-22: Craig Risien. Initial Code
        2014-05-27: Craig Risien. This function now looks for the light vs
                    dark frame measurements and only calculates nitrate
                    concentration based on the light frame measurements.
        2015-04-09: Russell Desiderio. CI is now implementing cal coeffs
                    by tiling in time, requiring coding changes. The
                    tiling includes the wllower and wlupper variables
                    when supplied by CI.
        2025-09-19: Andrew Reed. Updates for improved T,S,P corrections
                    following Plant et al 2023 and Sakamoto et al 2017

    Usage:

        NO3_conc = ts_corrected_nitrate(cal_temp, wl, eno3, eswa, di,
                                        dark_value, ctd_t, ctd_sp, data_in,
                                        frame_type, wllower, wlupper)

            where

        cal_temp = Calibration water temperature value
        wl = (256,) array of wavelength bins
        eno3 = (256,) array of wavelength-dependent nitrate
                extinction coefficients
        eswa = (256,) array of seawater extinction coefficients
        di = (256,) array of deionized water reference spectrum
        dark_value = (N,) array of dark average scalar value
        ctd_t = (N,) array of water temperature values from
                colocated CTD [deg C].
                (see 1341-00010_Data_Product_Spec_TEMPWAT)
        ctd_sp = (N,) array of practical salinity values from
                colocated CTD [unitless].
                (see 1341-00040_Data_Product_Spec_PRACSAL)
        ctd_p = (N,) array of ctd pressure in dbar
        data_in = (N x 256) array of nitrate measurement values
                from the UV absorption spectrum data product
                (L0 NITROPT) [unitless]
        NO3_conc = L2 Dissolved Nitrate Concentration, Temperature and
                Corrected (NITRTSC) [uM]
        frame_type = (N,) array of Frame type, either a light or dark
                measurement. This function only uses the data from light
                frame measurements.
        wllower = Lower wavelength limit for spectra fit.
                  From DPS: 217 nm (1-cm pathlength probe tip) or
                            220 nm (4-cm pathlength probe tip)
        wlupper = Upper wavelength limit for spectra fit.
                  From DPS: 240 nm (1-cm pathlength probe tip) or
                            245 nm (4-cm pathlength probe tip)
    Notes:

    References:

        OOI (2014). Data Product Specification for NUTNR Data Products.
            Document Control Number 1341-00620.
            https://alfresco.oceanobservatories.org/ (See: Company Home >>
            OOI >> Controlled >> 1000 System Level >>
            1341-00620_Data_Product_Spec_NUTNR_OOI.pdf)
        Johnson, K. S., and L. J. Coletti. 2002. In situ ultraviolet
            spectrophotometry for high resolution and long-term monitoring
            of nitrate, bromide and bisulfide in the ocean. Deep-Sea Res.
            I 49:1291-1305
        Sakamoto, C.M., K.S. Johnson, and L.J. Coletti (2009). Improved
            algorithm for the computation of nitrate concentrations in
            seawater using an in situ ultraviolet spectrophotometer.
            Limnology and Oceanography: Methods 7: 132-143
        Plant, J. N., Sakamoto, C. M., Johnson, K. S., Maurer, T. L., & Bif, M. B. (2023).
            Updated temperature correction for computing seawater nitrate with in situ 
            ultraviolet spectrophotometer and submersible ultraviolet nitrate analyzer 
            nitrate sensors. Limnology and Oceanography: Methods 21(10): 581–593. 
            https://doi.org/10.1002/lom3.10566
    """
    logger.info(nutnr)
    logger.info("Loading calibrations")
    cals = get_calibration_dataset(site)

    logger.info("Building calibration index (no broadcast)...")
    coeffs = broadcast_calibrations(cals, nutnr.time)
    cal_tables = coeffs["tables"]

    cal_temp = cal_tables["CC_cal_temp"]
    wl = cal_tables["CC_wl"]
    eno3 = cal_tables["CC_eno3"]
    eswa = cal_tables["CC_eswa"]
    di = cal_tables["CC_di"]
    wllower = cal_tables["CC_lower_wavelength_limit_for_spectra_fit"]
    wlupper = cal_tables["CC_upper_wavelength_limit_for_spectra_fit"]
    
    dark_value = nutnr.nutnr_dark_value_used_for_fit.values
    ctd_t = nutnr.sea_water_temperature.values
    ctd_sp = nutnr.sea_water_practical_salinity.values
    data_in = nutnr.spectral_channels.values
    frame_type = nutnr.frame_type.values
                               
    n_data_packets = data_in.shape[0]

    cal_index = coeffs["index"]

    # Ensure 2D (n_cal_rows x n_wavelengths) before indexing
    if wl.ndim == 1:
        wl = wl[np.newaxis, :]
        di = di[np.newaxis, :]
        eno3 = eno3[np.newaxis, :]
        eswa = eswa[np.newaxis, :]

    # Expand 1D cal arrays to n_data_packets using cal_index
    if np.ndim(wllower) == 0:
        wllower = np.full(n_data_packets, float(wllower))
    else:
        wllower = wllower[cal_index]
    if np.ndim(wlupper) == 0:
        wlupper = np.full(n_data_packets, float(wlupper))
    else:
        wlupper = wlupper[cal_index]
    if np.ndim(cal_temp) == 0:
        cal_temp = np.full(n_data_packets, float(cal_temp))
    else:
        cal_temp = cal_temp[cal_index]

    wl = wl[cal_index, :]
    di = di[cal_index, :]
    eno3 = eno3[cal_index, :]
    eswa = eswa[cal_index, :]

    c0 = 1.46380e-02
    c1 = 1.67660e-03
    c2 = 2.91898e-05
    c3 = -7.56395e-06
    c4 = 1.27353e-07

    NO3_conc = np.ones(n_data_packets)
    fitting_function = np.empty(n_data_packets, dtype=object)
    rmse = np.ones(n_data_packets)

    for i in range(0, n_data_packets):

        if frame_type[i] == 'SDB' or frame_type[i] == 'SDF' or frame_type[i] == "NDF":
            NO3_conc[i] = np.nan
            fitting_function[i] = np.nan
            rmse[i] = np.nan

        else:
            useindex = np.logical_and(wllower[i] <= wl[i, :], wl[i, :] <= wlupper[i])

            WL = wl[i, useindex]
            ENO3 = eno3[i, useindex]
            ESWA = eswa[i, useindex]
            DI = np.array(di[i, useindex], dtype='float64')
            SW = np.array(data_in[i, useindex], dtype='float64')

            SWcorr = SW - dark_value[i]
            with np.errstate(divide='ignore', invalid='ignore'):
                Absorbance = np.log10(DI / SWcorr)

            WL_prime = (WL - 210.0)
            f_prime = (c0 + c1 * WL_prime + c2 * WL_prime**2 + c3 * WL_prime**3 
                       + c4 * WL_prime**4)

            SWA_Ext_at_T = (ESWA * np.exp(f_prime * (ctd_t[i] - cal_temp[i])))

            A_SWA = ctd_sp[i] * SWA_Ext_at_T
            Acomp = np.array(Absorbance - A_SWA, ndmin=2).T

            subset_array_size = np.shape(ENO3)
            Ones = np.ones((subset_array_size[0],), dtype='float64') / 100
            M = np.vstack((ENO3, Ones, WL / 1000)).T

            C = np.dot(np.linalg.pinv(M), Acomp)

            NO3_conc[i] = C[0, 0]

            # Reconstruct the fitted absorbance from the solved coefficients
            Afit = np.dot(M, C).flatten()
            # Compute RMSE between the fitted and measured absorbance
            residuals = Acomp.flatten() - Afit
            rmse[i] = np.sqrt(np.mean(residuals**2))
            # Store the fitting function as the coefficients array [NO3, baseline_const, slope]
            fitting_function[i] = C.flatten()

    coords = {"time": nutnr["time"]}

    logger.info("Wrapping outputs into DataArrays")

    return (
        xr.DataArray(NO3_conc,        coords=coords, dims=["time"], name="dissolved_nitrate"),
        xr.DataArray(fitting_function, coords=coords, dims=["time"], name="nutnr_fitting_function"),
        xr.DataArray(rmse,            coords=coords, dims=["time"], name="nutnr_rmse"),
    )