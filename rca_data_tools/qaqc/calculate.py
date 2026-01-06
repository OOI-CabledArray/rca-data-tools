"""calculate.py

This module contains code for creating data products
useful to the QA/QC process that are not served
through OOI/M2M.

"""

import ast
import xarray as xr
import numpy as np
from typing import Dict
from rca_data_tools.qaqc.qartod import loadStagedQARTOD
from rca_data_tools.qaqc.constants import all_configs_dict, variable_dict, qartod_skip_dict
from rca_data_tools.qaqc.utils import select_logger

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
    apg = optaa["optical_absorption"]
    m412 = np.nanargmin(np.abs(optaa["wavelength_a"].values[:] - 412.0))
    m440 = np.nanargmin(np.abs(optaa["wavelength_a"].values[:] - 440.0))
    m490 = np.nanargmin(np.abs(optaa["wavelength_a"].values[:] - 490.0))
    m530 = np.nanargmin(np.abs(optaa["wavelength_a"].values[:] - 530.0))
    m676 = np.nanargmin(np.abs(optaa["wavelength_a"].values[:] - 676.0))

    optaa["ratio_cdom"] = apg[:, m412] / apg[:, m440]
    optaa["ratio_carotenoids"] = apg[:, m490] / apg[:, m440]
    optaa["ratio_phycobilins"] = apg[:, m530] / apg[:, m440]
    optaa["ratio_qband"] = apg[:, m676] / apg[:, m440]

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
    m650 = np.argmin(
        np.abs(coeffs["a_wavelengths"] - 650.0)
    )  # find the closest wavelength to 650 nm
    m676 = np.argmin(
        np.abs(coeffs["a_wavelengths"] - 676.0)
    )  # find the closest wavelength to 676 nm
    m715 = np.argmin(
        np.abs(coeffs["a_wavelengths"] - 715.0)
    )  # find the closest wavelength to 715 nm
    apg = optaa["apg_ts_s"]
    abl = (
        (
            apg[:, m715 - 1 : m715 + 2].median(axis=1)
            - apg[:, m650 - 1 : m650 + 2].median(axis=1)
        )
        / (715 - 650)
    ) * (676 - 650) + apg[:, m650 - 1 : m650 + 2].median(axis=1)  # interpolate to 676 nm
    aphi = apg[:, m676 - 1 : m676 + 2].median(axis=1) - abl
    optaa["estimated_chlorophyll"] = aphi / chl_line_height

    # estimate the POC concentration from the attenuation at 660 nm, from Cetinic et al., 2012 and references therein
    # (doi:10.4319/lom.2012.10.415)
    m660 = np.argmin(
        np.abs(coeffs["c_wavelengths"] - 660.0)
    )  # find the closest wavelength to 660 nm
    cpg = optaa["cpg_ts"]
    optaa["estimated_poc"] = cpg[:, m660 - 1 : m660 + 2].median(axis=1) * 381

    return optaa


class QartodRunner:
    def __init__(
        self,
        refdes: str,
        param: str,
        da: xr.DataArray,
        qartod_ds: xr.Dataset,
        qc_flags: Dict,
    ):
        """
        parameters:
        refdes: str, reference designator of instrument
        param: str, name of data param of interest ie: 'ph_seawater' subset to time of interest
        da: xarray DataArray, actual data array of param of interest used to run qartod tests against tables
        qartod_ds: xarray Dataset, dataset of qartod test arrays and param of interest, may be helpful for persisting attrs
        qc_flags: dict of qartod flag mappings useful for generating array names ie:
            {
                'qartod_grossRange': {
                    'symbol': '+',
                    'param': '_qartod_executed_gross_range_test'
                },
                'qartod_climatology': {
                    'symbol': 'x',
                    'param': '_qartod_executed_climatology_test'
                },
                'qc': {
                    'symbol': 's',
                    'param': '_qc_summary_flag'
                }
            }
        self.qartod(): method that runs homebrew qartod tests and returns a dataset that can overwrite
        the qcDS generated by CI for when the data team wants to review new or preliminary qartod
        tables. This method will be triggered with '--homebrew-qartod' flag. And the resulting
        pngs will probably be stored in an alternate s3 bucket to be viewed with the staging frontend.

        loadStagedQARTOD(): function that loads qartod tables from RCA qartod staging repo and formats
        the tables so we can run homebrew tests in this plotting routine.

        NOTE as of 2025 profiling instruments have binned climatology tables and integrated gross range
        fixed instruments have fixed climatology and fixed gross range tables.
        """
        # TODO in the plot profile scatter loop this class is invoked in a loop which might cause github to api limit us
        # might need to seperate fetching tables so its run once at the beginning of a dashboard workflow
        self.refdes = refdes
        self.param = param
        self.da = da
        self.param_da = self.clean_param_data_array()
        self.pressure_param = self.get_pressure_param()
        self.qartod_ds = qartod_ds
        self.qc_flags = qc_flags
        self.qc_summary_da = self.get_qc_summary_da()
        self.table_type = self.determine_qartod_table_types()

        self.clim_dict, self.gross_dict = loadStagedQARTOD(refdes, param, self.table_type)

    def run_gross_range(self):
        param = self.param
        param_da = self.param_da
        qc_flags = self.qc_flags

        if self.table_type[1] is None:
            return None

        gross_da = xr.full_like(param_da, 1)  # qartod array which mirrors input array
        gross_da.name = f"{param}{qc_flags['qartod_grossRange']['param']}"

        if self.table_type[1] in [
            "fixed",
            "int",
        ]:  # gross range table is same format for fixed and integrated
            fail_low = self.gross_dict["qartod"]["gross_range_test"]["fail_span"][0]
            fail_high = self.gross_dict["qartod"]["gross_range_test"]["fail_span"][1]
            sus_low = self.gross_dict["qartod"]["gross_range_test"]["suspect_span"][0]
            sus_high = self.gross_dict["qartod"]["gross_range_test"]["suspect_span"][1]

            fail_mask = (param_da <= fail_low) | (param_da >= fail_high)
            suspect_mask = ((param_da <= sus_low) | (param_da >= sus_high)) & ~fail_mask

            gross_da = gross_da.where(~fail_mask, 4)
            gross_da = gross_da.where(~suspect_mask, 3)

            # gross_da.attrs = self.qartod_ds[gross_da.name].attrs
            return gross_da

    def run_climatology(self):
        da = self.da
        param = self.param
        param_da = self.param_da
        pressure_param = self.pressure_param
        qc_flags = self.qc_flags

        if self.table_type[0] is None:
            return None

        clim_da = xr.full_like(param_da, 1)  # qartod array which mirrors input array
        clim_da.name = f"{param}{qc_flags['qartod_climatology']['param']}"

        if self.table_type[0] == "fixed":
            for month_str in self.clim_dict.keys():
                month_int = ast.literal_eval(month_str)
                month_mask = da.time.dt.month == month_int

                sus_low = ast.literal_eval(self.clim_dict[month_str]["[0, 0]"])[0]
                sus_high = ast.literal_eval(self.clim_dict[month_str]["[0, 0]"])[1]
                logger.info(
                    f"Month: {month_int}, Suspect Low: {sus_low}, Suspect High: {sus_high}"
                )

                suspect_mask = ((param_da <= sus_low) | (param_da >= sus_high)) & month_mask
                # clim_da.attrs = self.qartod_ds[clim_da.name].attrs
                clim_da = clim_da.where(~suspect_mask, 3)

        elif self.table_type[0] == "binned":
            # TODO double check this logic # TODO how do we get pressure for profilers fixed depth
            for month_str in self.clim_dict.keys():
                month_int = ast.literal_eval(month_str)
                month_mask = da.time.dt.month == month_int

                for depth_range_str in self.clim_dict[month_str].keys():
                    depth_range = ast.literal_eval(depth_range_str)
                    depth_mask = (da[pressure_param] >= depth_range[0]) & (
                        da[pressure_param] < depth_range[1]
                    )

                    sus_low = ast.literal_eval(self.clim_dict[month_str][depth_range_str])[0]
                    sus_high = ast.literal_eval(self.clim_dict[month_str][depth_range_str])[1]
                    logger.info(
                        f"Month: {month_int}, Depth Range: {depth_range}, Suspect Low: {sus_low}, Suspect High: {sus_high}"
                    )

                    suspect_mask = (
                        ((param_da <= sus_low) | (param_da >= sus_high))
                        & month_mask
                        & depth_mask
                    )
                    clim_da = clim_da.where(~suspect_mask, 3)

        return clim_da

    def qartod(self):
        gross_da = self.run_gross_range()
        climatology_da = self.run_climatology()

        merge_vars = [self.param_da] # parameter we ran the qartod test on

        # if tests were skipped we won't return those arrays
        if gross_da is not None:
            merge_vars.append(gross_da)
        if climatology_da is not None:
            merge_vars.append(climatology_da)

        # merge pressure param if applicable
        if self.pressure_param:
            merge_vars.append(self.da[self.pressure_param])

        if self.qc_summary_da is not None:
            merge_vars.append(self.qc_summary_da)

        homebrew_qartod_ds = xr.merge(merge_vars)

        return homebrew_qartod_ds

    def get_pressure_param(self):
        pressure_vars = variable_dict["pressure"].strip('"').split(",")
        if isinstance(self.da, xr.Dataset):
            matching_vars = [var for var in pressure_vars if var in self.da.data_vars]
            if matching_vars:
                pressure_param = matching_vars[0]
                return pressure_param
            else:
                # if no pressure var in dataset
                return None

        else:
            # if single data array
            return None

    def clean_param_data_array(self):
        da = self.da
        param = self.param

        if isinstance(da, xr.Dataset):
            param_da = da[param]
        elif isinstance(da, xr.DataArray):
            param_da = da

        return param_da

    def get_qc_summary_da(self):
        try: 
            qc_summary_da = self.qartod_ds[f"{self.param}{self.qc_flags['qc']['param']}"]
        except KeyError:
            logger.info(f"QC summary data array not found for {self.refdes} {self.param}")
            qc_summary_da = None
        return qc_summary_da
    
    
    def determine_qartod_table_types(self):
        refdes = self.refdes

        if "FIXED" in all_configs_dict[refdes]["instrument"]:
            table_type = [
                "fixed",
                "fixed",
              ]  # (clim, gross) both fixed for fixed instruments
        elif "PROFILER" in all_configs_dict[refdes]["instrument"]:
            table_type = [
                "binned",
                "int",
              ]  # (clim, gross) gross range is integrated for profilers

        if self.param in qartod_skip_dict.keys():
            skip_tests = qartod_skip_dict[self.param]
            if "climatology" in skip_tests:
                table_type[0] = None # skip climatology test
            if "gross_range" in skip_tests:
                table_type[1] = None # skip gross range test

        return table_type

        

