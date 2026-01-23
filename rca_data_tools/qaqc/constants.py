import pandas as pd
import yaml
from pathlib import Path

# visual data constants
CAM_URL_DICT = {
    'RS01SUM2-MJ01B-05-CAMDSB103': 'https://rawdata.oceanobservatories.org/files/RS01SUM2/MJ01B/CAMDSB103/',
    'RS03INT1-MJ03C-05-CAMDSB303': 'https://rawdata.oceanobservatories.org/files/RS03INT1/MJ03C/CAMDSB303/',
    'RS03AXPS-PC03A-07-CAMDSC302': 'https://rawdata.oceanobservatories.org/files/RS03AXPS/PC03A/CAMDSC302/',
    'CE02SHBP-MJ01C-08-CAMDSB107': 'https://rawdata.oceanobservatories.org/files/CE02SHBP/MJ01C/CAMDSB107/',
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

all_configs_dict = {**sites_dict, **stage2_dict, **stage3_dict}

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

qartod_skip_dict = yaml.safe_load(open(PARAMS_DIR.joinpath("qartod_skip.yaml"))) 

COMPUTE_EXCEPTIONS = yaml.safe_load(open(PARAMS_DIR.joinpath("compute_exceptions.yaml")))
