"""
Streams that require more compute resources on AWS than 2 vcpu 
and 16 gb. (Those are the defaults associated with the prefect 2
workpool.)

"""
import pandas as pd
from pathlib import Path

COMPUTE_EXCEPTIONS = {
    # spkira
    'CE04OSPS-SF01B-3D-SPKIRA102':{
        '365': '4vcpu_30gb',
        '30': '4vcpu_30gb',
        '7': '4vcpu_30gb',
        '1': '4vcpu_30gb',
    },
    'RS01SBPS-SF01A-3D-SPKIRA101':{
        '365': '8vcpu_60gb',
        '30': '8vcpu_60gb',
        '7': '8vcpu_60gb',
        '1': '8vcpu_60gb',
    },
    'RS03AXPS-SF03A-3D-SPKIRA301':{
        '365': '8vcpu_60gb',
        '30': '8vcpu_60gb',
        '7': '8vcpu_60gb',
        '1': '8vcpu_60gb',
    },
    # velptd
    'RS01SBPS-SF01A-4B-VELPTD102':{
        '365': '4vcpu_30gb',
    },
    'RS03AXPS-SF03A-4B-VELPTD302':{
        '365': '4vcpu_30gb',
    },
    # ctdbpo
    'CE04OSBP-LJ01C-06-CTDBPO108':{
        '365': '4vcpu_30gb',
    },
    # ctdpfa
    'CE04OSPS-SF01B-2A-CTDPFA107':{
        '365': '4vcpu_30gb',
    },
    # adcp
    'RS01SLBS-LJ01A-10-ADCPTE101':{
        '30': '8vcpu_60gb',
    },
    'RS01SUM2-MJ01B-12-ADCPSK101':{
        '30': '8vcpu_60gb',
    },
    'CE02SHBP-LJ01D-05-ADCPTB104':{
        '30': '8vcpu_60gb',
        '7': '4vcpu_30gb',
    },
    'RS03AXBS-LJ03A-10-ADCPTE303':{
        '30': '4vcpu_30gb',
    },
    # parad
    'RS03AXPS-SF03A-3C-PARADA301':{
        '365': '4vcpu_30gb',
        '30': '4vcpu_30gb',
        '7': '4vcpu_30gb',
        '1': '4vcpu_30gb',
    },
    'RS01SBPS-SF01A-3C-PARADA101':{
        '365': '4vcpu_30gb',
        '30': '4vcpu_30gb',
        '7': '4vcpu_30gb',
        '1': '4vcpu_30gb',
    },
}


CAM_URL_DICT = {
    'RS01SUM2-MJ01B-05-CAMDSB103': 'https://rawdata.oceanobservatories.org/files/RS01SUM2/MJ01B/CAMDSB103_10.33.7.5/',
    'RS03INT1-MJ03C-05-CAMDSB303': 'https://rawdata.oceanobservatories.org/files/RS03INT1/MJ03C/CAMDSB303_10.31.8.5/',
    'RS03AXPS-PC03A-07-CAMDSC302': 'https://rawdata.oceanobservatories.org/files/RS03AXPS/PC03A/CAMDSC302_10.31.3.146/',
    'CE02SHBP-MJ01C-08-CAMDSB107': 'https://rawdata.oceanobservatories.org/files/CE02SHBP/MJ01C/CAMDSB107_10.33.13.8/',
    'CE04OSBP-LV01C-06-CAMDSB106': 'https://rawdata.oceanobservatories.org/files/CE04OSBP/LV01C/CAMDSB106_10.33.9.6/',
    'RS01SBPS-PC01A-07-CAMDSC102': 'https://rawdata.oceanobservatories.org/files/RS01SBPS/PC01A/CAMDSC102_10.33.3.146/',
    }
N_EXPECTED_IMGS = 145

S3_BUCKET = 'ooi-rca-qaqc-prod'
#SPAN_DICT = {'1': 'day', '7': 'week', '30': 'month', '365': 'year'}

# CSV config constants 

HERE = Path(__file__).parent.absolute()
PARAMS_DIR = HERE.joinpath('params')
PLOT_DIR = Path('QAQC_plots')

SPAN_DICT = {
    '1': 'day',
    '7': 'week',
    '30': 'month',
    '365': 'year',
    '0': 'deploy',
}

# create dictionary of sites key for filePrefix, nearestNeighbors
sites_dict = (
    pd.read_csv(PARAMS_DIR.joinpath('sitesDictionary.csv'))
    .set_index('refDes')
    .T.to_dict('series')
)

# TODO different stage dictonaries now need to be piped, and probably renamed for clarity
stage3_dict = (
    pd.read_csv(PARAMS_DIR.joinpath('stage3Dictionary.csv'))
    .set_index('refDes')
    .T.to_dict('series')
)

# create dictionary of parameter vs variable Name
variable_dict = pd.read_csv(
    PARAMS_DIR.joinpath('variableMap.csv'), index_col=0, squeeze=True
).to_dict()

# create dictionary of instrumet key for plot parameters
instrument_dict = (
    pd.read_csv(PARAMS_DIR.joinpath('plotParameters.csv'))
    .set_index('instrument')
    .T.to_dict('series')
)

# create dictionary of variable parameters for plotting
variable_paramDict = (
    pd.read_csv(PARAMS_DIR.joinpath('variableParameters.csv'))
    .set_index('variable')
    .T.to_dict('series')
)

# create dictionary of multi-parameter instrumet variables
multiParameter_dict = (
    pd.read_csv(PARAMS_DIR.joinpath('multiParameters.csv'))
    .set_index('instrument')
    .T.to_dict('series')
)

# create dictionary of local parameter ranges for each site
localRange_dict = (
    pd.read_csv(PARAMS_DIR.joinpath('localRanges.csv'))
    .set_index('refDes')
    .T.to_dict('series')
)

# create a dictonary of sites with partially active coordinates for current deployment
deployedRange_dict = (
    pd.read_csv(PARAMS_DIR.joinpath('deployedRanges.csv'))
    .set_index('refDes')
    .T.to_dict('series')
)

plotDir = str(PLOT_DIR) + '/'