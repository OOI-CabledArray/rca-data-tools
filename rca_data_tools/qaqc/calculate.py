# -*- coding: utf-8 -*-
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

