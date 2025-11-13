"""calculate.py

This module contains code for creating data products 
useful to the QA/QC process that are not served
through OOI/M2M.

"""

import numpy as np

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
    volts = 5. * traw / 65535.

    # calculate the resistance of the thermistor
    res = 10000. * volts / (4.516 - volts)

    # convert resistance to temperature
    a = 0.00093135
    b = 0.000221631
    c = 0.000000125741

    log_res = np.log(res)
    degC = (1. / (a + b * log_res + c * log_res**3)) - 273.15
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

    degC = a * np.power(traw.astype('O'), 3) + b * np.power(traw.astype('O'), 2) + c * traw + d
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
    return depth




def opt_calculate_ratios(optaa):
    """
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

    :param optaa: xarray dataset with the scatter corrected absorbance data.
    :return optaa: xarray dataset with the estimates for pigment ratios added.
    """
    apg = optaa['optical_absorption']
    m412 = np.nanargmin(np.abs(optaa['wavelength_a'].values[:] - 412.0))
    m440 = np.nanargmin(np.abs(optaa['wavelength_a'].values[:] - 440.0))
    m490 = np.nanargmin(np.abs(optaa['wavelength_a'].values[:] - 490.0))
    m530 = np.nanargmin(np.abs(optaa['wavelength_a'].values[:] - 530.0))
    m676 = np.nanargmin(np.abs(optaa['wavelength_a'].values[:] - 676.0))

    optaa['ratio_cdom'] = apg[:, m412] / apg[:, m440]
    optaa['ratio_carotenoids'] = apg[:, m490] / apg[:, m440]
    optaa['ratio_phycobilins'] = apg[:, m530] / apg[:, m440]
    optaa['ratio_qband'] = apg[:, m676] / apg[:, m440]

    return optaa



def opt_estimate_chl_poc(optaa, coeffs, chl_line_height=0.020):
    """
    Derive estimates of Chlorophyll-a and particulate organic carbon (POC)
    concentrations from the temperature, salinity and scatter corrected
    absorption and beam attenuation data.

    :param optaa: xarray dataset with the scatter corrected absorbance data.
    :param coeffs: Factory calibration coefficients in a dictionary structure
    :param chl_line_height: Extinction coefficient for estimating the
        chlorophyll concentration. This value may vary regionally and/or
        seasonally. A default value of 0.020 is used if one is not entered,
        but users may to adjust this based on cross-comparisons with other
        measures of chlorophyll
    :return optaa: xarray dataset with the estimates for chlorophyll and POC
        concentrations added.
    """
    # use the standard chlorophyll line height estimation with an extinction coefficient of 0.020,
    # from Roesler and Barnard, 2013 (doi:10.4319/lom.2013.11.483)
    m650 = np.argmin(np.abs(coeffs['a_wavelengths'] - 650.0))  # find the closest wavelength to 650 nm
    m676 = np.argmin(np.abs(coeffs['a_wavelengths'] - 676.0))  # find the closest wavelength to 676 nm
    m715 = np.argmin(np.abs(coeffs['a_wavelengths'] - 715.0))  # find the closest wavelength to 715 nm
    apg = optaa['apg_ts_s']
    abl = ((apg[:, m715-1:m715+2].median(axis=1) - apg[:, m650-1:m650+2].median(axis=1)) /
           (715 - 650)) * (676 - 650) + apg[:, m650-1:m650+2].median(axis=1)  # interpolate to 676 nm
    aphi = apg[:, m676-1:m676+2].median(axis=1) - abl
    optaa['estimated_chlorophyll'] = aphi / chl_line_height

    # estimate the POC concentration from the attenuation at 660 nm, from Cetinic et al., 2012 and references therein
    # (doi:10.4319/lom.2012.10.415)
    m660 = np.argmin(np.abs(coeffs['c_wavelengths'] - 660.0))  # find the closest wavelength to 660 nm
    cpg = optaa['cpg_ts']
    optaa['estimated_poc'] = cpg[:, m660-1:m660+2].median(axis=1) * 381

    return optaa


class QartodRunner:
    def __init__(
        self,
        ds,
        qartod_ds, # passsed from dashboard.retrieve_qc to model what qartod output arrays should look like 
        qc_flags, # flags ie {'qartod_grossRange': {'symbol': '+', 'param': '_qartod_executed_gross_range_test'}, 'qartod_climatology': {'symbol': 'x', 'param': '_qartod_executed_climatology_test'}, 'qc': {'symbol': 's', 'param': '_qc_summary_flag'

    ):
        self.ds = ds
        self.qartod_ds = qartod_ds
        self.qc_flags = qc_flags

        #pseudocode 
        # take in param ds and live qartod ds 
        # read in qartod tables from RCA qartod staging 
        # run the actual tests with those staging values from qartod-staging 
        # output an array that looks like qartod_ds but maybe without qc flags vars (just qartod)
        # then return a qartod_ds da with the homebrewed qartod results, 
        # plop in an alternate bucket? 
        # view with alternate deployment of frontend that requires VPN #TODO

        
        pass