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
    'CE04OSPS-SF01B-4B-VELPTD106':{
        '365': '4vcpu_30gb',
    },
    'RS03AXPS-SF03A-4B-VELPTD302':{
        '365': '4vcpu_30gb',
    },
    # vel3d
    'RS01SLBS-MJ01A-12-VEL3DB101':{
        '365': '4vcpu_30gb',
    },
    'RS01SUM1-LJ01B-12-VEL3DB104':{
        '365': '4vcpu_30gb',
    },
    'CE04OSBP-LJ01C-07-VEL3DC107':{
        '365': '16vcpu_104gb',
        '30': '16vcpu_88gb',
        '7': '16vcpu_88gb',
        '1': '16vcpu_88gb',
    },
    'CE02SHBP-LJ01D-07-VEL3DC108':{
        '365': '16vcpu_96gb',
        '30': '16vcpu_80gb',
        '7': '16vcpu_80gb',
        '1': '16vcpu_80gb',
    },
    'RS03AXBS-MJ03A-12-VEL3DB301':{
        '365': '4vcpu_30gb',
    },
    'RS03INT2-MJ03D-12-VEL3DB304':{
        '365': '8vcpu_60gb',
    },
    # ctd
    'CE04OSBP-LJ01C-06-CTDBPO108':{
        '365': '4vcpu_30gb',
    },
    'RS01SBPS-PC01A-4A-CTDPFA103':{
        '365': '4vcpu_30gb',
    },
    'CE04OSPS-SF01B-2A-CTDPFA107':{
        '365': '4vcpu_30gb',
    },
    'RS03AXPS-PC03A-4A-CTDPFA303':{
        '365': '4vcpu_30gb',
    },
    'RS03AXBS-LJ03A-12-CTDPFB301':{
        '365': '4vcpu_30gb',
    },
    'RS01SLBS-LJ01A-12-CTDPFB101':{
        '365': '4vcpu_30gb',
    },
    'CE02SHBP-LJ01D-06-CTDBPN106':{
        '365': '4vcpu_30gb',
    },
    # adcp
    'RS01SLBS-LJ01A-10-ADCPTE101':{
        '30': '4vcpu_30gb',
        '0': '4vcpu_30gb',
    },
    'RS01SUM2-MJ01B-12-ADCPSK101':{
        '365': '4vcpu_30gb',
        '30': '4vcpu_30gb',
        '0': '4vcpu_30gb',
    },
    'CE02SHBP-LJ01D-05-ADCPTB104':{
        '365': '4vcpu_30gb',
        '30': '4vcpu_30gb',
        '7': '4vcpu_30gb',
        '1': '4vcpu_30gb',
        '0': '8vcpu_60gb',
    },
    'RS03AXBS-LJ03A-10-ADCPTE303':{
        '365': '4vcpu_30gb',
        '30': '4vcpu_30gb',
        '0': '4vcpu_30gb',
    },
    'CE04OSBP-LJ01C-05-ADCPSI103':{
        '365': '4vcpu_30gb',
        '30': '4vcpu_30gb',
        '0': '4vcpu_30gb',
    },
    # vadcp
    'RS01SBPS-PC01A-06-VADCPA101':{
        '365': '4vcpu_30gb',
        '0': '4vcpu_30gb',
    },
    'RS03AXPS-PC03A-06-VADCPA301':{
        '365': '4vcpu_30gb',
        '0': '4vcpu_30gb',
    },
    # parad
    'RS03AXPS-SF03A-3C-PARADA301':{
        '365': '8vcpu_60gb',
        '30': '8vcpu_60gb',
        '7': '8vcpu_60gb',
        '1': '8vcpu_60gb',
    },
    'RS01SBPS-SF01A-3C-PARADA101':{
        '365': '8vcpu_60gb',
        '30': '8vcpu_60gb',
        '7': '8vcpu_60gb',
        '1': '8vcpu_60gb',
    },
    # tmpsf
    'RS03ASHS-MJ03B-07-TMPSFA301':{
        '365': '4vcpu_30gb',
    },
}

# visual data constants
CAM_URL_DICT = {
    'RS01SUM2-MJ01B-05-CAMDSB103': 'https://rawdata.oceanobservatories.org/files/RS01SUM2/MJ01B/CAMDSB103/',
    'RS03INT1-MJ03C-05-CAMDSB303': 'https://rawdata.oceanobservatories.org/files/RS03INT1/MJ03C/CAMDSB303/',
    'RS03AXPS-PC03A-07-CAMDSC302': 'https://rawdata.oceanobservatories.org/files/RS03AXPS/PC03A/CAMDSC302/',
    'CE02SHBP-MJ01C-08-CAMDSB107': 'https://rawdata.oceanobservatories.org/files/CE02SHBP/MJ01C/CAMDSB107_10.33.13.8/',
    'CE04OSBP-LV01C-06-CAMDSB106': 'https://rawdata.oceanobservatories.org/files/CE04OSBP/LV01C/CAMDSB106/',
    'RS01SBPS-PC01A-07-CAMDSC102': 'https://rawdata.oceanobservatories.org/files/RS01SBPS/PC01A/CAMDSC102/',
    'RS03ASHS-PN03B-06-CAMHDA301': 'https://rawdata.oceanobservatories.org/files/RS03ASHS/PN03B/CAMHDA301/',
    }
N_EXPECTED_IMGS = 145

S3_BUCKET = 'ooi-rca-qaqc-prod'

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

CAM_SPANS = {
    '7': 'week', 
    '30': 'month', 
    '365': 'year',
    '0': 'deploy'
}

THROTTLE_SPANS = {
    '1': 'day',
    '7': 'week',
}

statusColors = {
    'OPERATIONAL': 'green',
    'FAILED': 'red',
    'TROUBLESHOOTING': 'red',
    'RECOVERED': 'blue',
    'PARTIALLY_FUNCTIONAL': 'red',
    'OFFLINE': 'blue',
    'UNCABLED': 'blue',
    'DATA_QUALITY': 'red',
    'NOT_DEPLOYED': 'blue'
}

# create dictionary of sites key for filePrefix, nearestNeighbors
sites_dict = (
    pd.read_csv(PARAMS_DIR.joinpath('sitesDictionary.csv'))
    .set_index('refDes')
    .T.to_dict('series')
)

stage2_dict = (
    pd.read_csv(PARAMS_DIR.joinpath('stage2Dictionary.csv'))
    .set_index('refDes')
    .T.to_dict('series')
)

stage3_dict = (
    pd.read_csv(PARAMS_DIR.joinpath('stage3Dictionary.csv'))
    .set_index('refDes')
    .T.to_dict('series')
)

# create dictionary of parameter vs variable Name
variable_dict = pd.read_csv(PARAMS_DIR.joinpath('variableMap.csv'), index_col=0).iloc[:, 0].to_dict()

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

# create a dictonary of auxilliary parameters to be calculated
calculate_dict = (
    pd.read_csv(PARAMS_DIR.joinpath('calculateParameters.csv'))
    .set_index('refDes')
    .T.to_dict('series')
) 

# create a dictonary of calculations and inputs as executable strings
calculateStrings_dict = (
    pd.read_csv(PARAMS_DIR.joinpath('calculateStrings.csv'))
    .set_index('calculation')
    .T.to_dict()
)

# create a dictonary of relevant discrete sample types for each variable
discreteSample_dict = (
    pd.read_csv(PARAMS_DIR.joinpath('discreteMap.csv'))
    .set_index('variable')
    .T.to_dict()
)

plotDir = str(PLOT_DIR) + '/'
