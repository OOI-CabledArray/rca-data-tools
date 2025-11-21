import pytest
import pickle
from datetime import datetime
import pandas as pd
import xarray as xr

from rca_data_tools.qaqc.dashboard import plotProfilesScatter, loadProfiles

site = 'RS01SBPS-SF01A-4F-PCO2WA101'
Xparam = 'pco2_seawater'
param = 'pco2'
press_param = 'int_ctd_pressure'
param_data = xr.open_dataset('./tests/toy_data/PCO2WA101_paramData.nc')
plot_title = 'RS01SBPS-SF01A-4F-PCO2WA101 pco2'
time_ref = datetime(2025, 7, 30, 0, 0)
y_min = 0 
y_max = 200
profile_paramMin = 200.0
profile_paramMax = 1200.0
profile_paramMin_local = 30
profile_paramMax_local = 1400
fileName_base = 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2'
overlayData_anno = pickle.load(open('./tests/toy_data/PCO2WA101_overlayData_anno.pickle', 'rb'))
overlayData_clim = pickle.load(open('./tests/toy_data/PCO2WA101_overlayData_clim.pickle', 'rb'))
overlayData_disc = pd.DataFrame()
overlayData_flag = xr.open_dataset('./tests/toy_data/PCO2WA101_overlayData_flag.nc')
overlayData_near = {}
span = '1'
span_string = 'day'
#profile_list = pd.read_csv('./tests/toy_data/profile_list.csv') # this may need to be static
profile_list = loadProfiles(site) # this is dynamic and could cause problems with the test, may be at root of index error
status_dict = {'RS01SBPS-SF01A-4F-PCO2WA101': 'OPERATIONAL'}
site = 'RS01SBPS-SF01A-4F-PCO2WA101'

expected_fileNameList = ['QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_none_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_anno_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_clim_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_flag_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_none_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_anno_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_clim_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_flag_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_none_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_anno_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_clim_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_flag_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_none_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_anno_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_clim_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_flag_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_none_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_anno_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_clim_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_flag_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_none_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_anno_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_clim_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_flag_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_none_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_anno_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_clim_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_flag_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_none_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_anno_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_clim_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_flag_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_none_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_anno_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_clim_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_flag_local.png']

@pytest.mark.parametrize(
    "test_timeRef",
    [
        datetime(2025, 7, 30, 0, 0), # might need to make this dynamic to the current day which could also catch above error referenced above
        datetime(2025, 7, 29, 0, 0), 
    ]
)
def test_plotProfilesScatter(test_timeRef):
    file_name_list = plotProfilesScatter(
        Xparam=Xparam,
        param=param,
        press_param=press_param,
        param_data=param_data,
        plot_title=plot_title,
        time_ref=test_timeRef,
        y_min=y_min,
        y_max=y_max,
        profile_paramMin=profile_paramMin,
        profile_paramMax=profile_paramMax,
        profile_paramMin_local=profile_paramMin_local,
        profile_paramMax_local=profile_paramMax_local,
        fileName_base=fileName_base,
        overlayData_anno=overlayData_anno,
        overlayData_clim=overlayData_clim,
        overlayData_disc=overlayData_disc,
        overlayData_flag=overlayData_flag,
        overlayData_near=overlayData_near,
        span=span,
        span_string=span_string,
        profile_list=profile_list,
        status_dict=status_dict,
        site=site
    )

    assert file_name_list == expected_fileNameList