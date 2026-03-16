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

### OPTAA functions

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