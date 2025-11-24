import pytest
import pickle
from datetime import datetime
import pandas as pd
import xarray as xr

from rca_data_tools.qaqc.dashboard import plotProfilesScatter, loadProfiles

site = 'RS01SBPS-SF01A-4F-PCO2WA101'
Xparam = 'pco2_seawater'
param = 'pco2'
pressParam = 'int_ctd_pressure'
paramData = xr.open_dataset('./tests/toy_data/PCO2WA101_paramData.nc')
plotTitle = 'RS01SBPS-SF01A-4F-PCO2WA101 pco2'
timeRef = datetime(2025, 7, 30, 0, 0)
yMin = 0 
yMax = 200
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
spanString = 'day'
#profileList = pd.read_csv('./tests/toy_data/profileList.csv') # this may need to be static
profileList = loadProfiles(site) # this is dynamic and could cause problems with the test, may be at root of index error
statusDict = {'RS01SBPS-SF01A-4F-PCO2WA101': 'OPERATIONAL'}
site = 'RS01SBPS-SF01A-4F-PCO2WA101'
homebrew_qartod = False

expected_fileNameList = ['QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_none_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_anno_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_clim_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_flag_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_none_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_anno_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_clim_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_flag_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_none_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_anno_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_clim_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_000profile_day_flag_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_none_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_anno_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_clim_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_flag_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_none_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_anno_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_clim_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_flag_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_none_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_anno_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_clim_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_001profile_day_flag_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_none_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_anno_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_clim_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_flag_full.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_none_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_anno_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_clim_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_flag_standard.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_none_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_anno_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_clim_local.png', 'QAQC_plots/RS01SBPS-SF01A-4F-PCO2WA101_pco2_002profile_day_flag_local.png']

@pytest.mark.parametrize(
    "test_timeRef",
    [
        datetime(2025, 7, 30, 0, 0), # might need to make this dynamic to the current day which could also catch above error referenced above
        datetime(2025, 7, 29, 0, 0), 
    ]
)
def test_plotProfilesScatter(test_timeRef):
    fileNameList = plotProfilesScatter(
        Xparam=Xparam,
        param=param,
        pressParam=pressParam,
        paramData=paramData,
        plotTitle=plotTitle,
        timeRef=test_timeRef,
        yMin=yMin,
        yMax=yMax,
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
        spanString=spanString,
        profileList=profileList,
        statusDict=statusDict,
        site=site,
        homebrew_qartod=homebrew_qartod
    )

    assert fileNameList == expected_fileNameList