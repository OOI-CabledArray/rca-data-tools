"""
Streams that require more compute resources on AWS than 2 vcpu 
and 16 gb. (Those are the defaults associated with the prefect 2
workpool.)

"""

COMPUTE_EXCEPTIONS = {
    # spkira
    'CE04OSPS-SF01B-3D-SPKIRA102':{
        '365': '4vcpu_30gb',
        '30': '4vcpu_30gb',
        '7': '4vcpu_30gb',
        '1': '4vcpu_30gb',
    },
    'RS01SBPS-SF01A-3D-SPKIRA101':{
        '365': '4vcpu_30gb',
        '30': '4vcpu_30gb',
        '7': '4vcpu_30gb',
        '1': '4vcpu_30gb',
    },
    'RS03AXPS-SF03A-3D-SPKIRA301':{
        '365': '4vcpu_30gb',
        '30': '4vcpu_30gb',
        '7': '4vcpu_30gb',
        '1': '4vcpu_30gb',
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
SPAN_DICT = {'1': 'day', '7': 'week', '30': 'month', '365': 'year'}