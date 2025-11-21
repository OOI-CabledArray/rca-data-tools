"""notes.py

This module contains code for reading HITL notes from
google spreadsheet and converting them to a set of
csv files to be displayed in the QAQC Dashboard.

"""

import os
from datetime import datetime
import click
import gspread
import pandas as pd
import fsspec
from pathlib import Path
from loguru import logger

from .pipeline import S3_BUCKET

HITL_NOTES_DIR = Path('HITL_notes')


def sync_s3(s3_bucket, storage_options={}):
    fmap = fsspec.get_mapper(
        f"{s3_bucket}/{HITL_NOTES_DIR.name}", **storage_options
    )
    if fmap.fs.isdir(fmap.root):
        fmap.fs.rm(fmap.root, recursive=True)
    fmap.fs.put(str(HITL_NOTES_DIR.absolute()), fmap.root, recursive=True)


def fetch_creds(service_json_path: str = ''):
    if service_json_path:
        if service_json_path.startswith("s3://"):
            JSON_PATH = fsspec.get_mapper(service_json_path)
        else:
            JSON_PATH = Path(service_json_path)
    else:
        raise ValueError("Please Provide A Service Account Credential")
    GSPREAD_DIR = Path(os.path.expanduser("~"), ".config", "gspread")
    GSPREAD_DIR.mkdir(exist_ok=True)
    GSPREAD_JSON = GSPREAD_DIR.joinpath("service_account.json")

    if isinstance(JSON_PATH, fsspec.FSMap):
        with JSON_PATH.fs.open(JSON_PATH.root, 'rb') as f:
            GSPREAD_JSON.write_bytes(f.read())
    else:
        with open(JSON_PATH, 'rb') as f:
            GSPREAD_JSON.write_bytes(f.read())


def read_logs() -> pd.DataFrame:
    gc = gspread.service_account()
    wks = gc.open("HITL Data QA/QC Log")
    df_HITL_list = []
    for ws in wks.worksheets():
        df = pd.DataFrame(ws.get_all_records())
        for col in df.columns:
            if 'Unnamed' in col:
                del df[col]
        df_HITL_list.append(df.transpose())
    df_HITL = pd.concat(df_HITL_list)
    return df_HITL.apply(lambda x: x.str.replace(',', '.'))


def generate_tables(df_HITL: pd.DataFrame) -> None:
    # Generate HITL status tables:
    # by Stage:
    plot_pages = {}
    plot_pages['Stage1'] = [
        'ADCP',
        'BOTPT',
        'CTD',
        'DOFSTA',
        'DOSTA',
        'FLCDR',
        'FLORT',
        'FLNTU',
        'FLOR',
        'NUTNR',
        'PARAD',
        'PHSEN',
        'PCO2W',
        'SPKIR',
        'VELPT',
    ]

    plot_pages['Stage2'] = [
        'CAMHD',
        'OPTAA',
        'PREST',
        'THSPH',
        'TMPSF',
        'TRHPH',
        'VEL3D',
        'ZPLSC',
    ]
    plot_pages['Stage3'] = [
        'CAMDS',
        'HPIES',
        'HYDBB',
        'HYDLF',
        'MASSP',
        'OBSBB',
        'OBSSP',
    ]
    plot_pages['Stage4'] = ['FLOBNC', 'FLOBNM', 'OSMOIA', 'PPS', 'RAS', 'D1000']

    # by Site:
    plot_pages['Sites'] = [
        'CEO2SHBP',
        'CE04OSBP',
        'CE04OSPD',
        'CE04OSPS',
        'RS01SBPD',
        'RS01SBPS',
        'RS01SLBS',
        'RS01SUM1',
        'RS01SUM2',
        'RS03AXBS',
        'RS03AXPD',
        'RS03AXPS',
        'RS03INT1',
        'RS03INT2',
        'RS03CCAL',
        'RS03ECAL',
        'RS03ASHS',
    ]

    # by Platform
    plot_pages['Platforms'] = {
        'BEP': ['BP'],
        'Deep-Profiler': ['DP0'],
        'Shallow-Profiler': ['SF0'],
        'Shallow-Profiler-200m_Platform': ['PC0'],
        'Seafloor': [
            'SLBS',
            'SUM1',
            'SUM2',
            'AXBS',
            'INT1',
            'INT2',
            'CCAL',
            'ECAL',
            'ASHS',
        ],
    }
    for page in plot_pages.keys():
        for item in plot_pages[page]:
            if any(ele in page for ele in ['Stage', 'Sites']):
                df_logList = df_HITL[df_HITL.index.str.contains(item)]
            elif 'Platforms' in page:
                df_logList = df_HITL[
                    df_HITL.index.str.contains('|'.join(plot_pages[page][item]))
                ]
            if not df_logList.empty:
                # only keep the first column (most recent note)
                df_logList = df_logList.iloc[:, 0]
                csv_table = df_logList.to_csv(header=False)
                with open(
                    HITL_NOTES_DIR.joinpath(f"HITL_{page}_{item}.csv"), "w"
                ) as f:
                    f.write(csv_table)
    status_list = ['Watchlist', 'Failed', 'Harvest', 'Plotting', 'Pending']
    for status in status_list:
        df_statusList = df_HITL[df_HITL[0].str.contains(status,case=False)]
        if not df_statusList.empty:
            df_statusList = df_statusList.iloc[:, 0]
            csv_table = df_statusList.to_csv(header=False)
            with open(
                HITL_NOTES_DIR.joinpath(f"HITL_Status_{status}.csv"), "w"
            ) as f:
                f.write(csv_table)


@click.command()
@click.option('--service-json-path', type=str, required=True, help='Path to the service JSON file')
@click.option('--s3-sync', is_flag=True, help='Enable S3 sync')
@click.option('--bucket', default=S3_BUCKET, help='Bucket name for cloud index if not default. (ie staging/test buckets)')
def main(service_json_path, s3_sync, bucket):
    logger.add("logfile_generate_tables_{time}.log")
    logger.info('HITL notes generation initiated')
    fetch_creds(service_json_path)
    now = datetime.utcnow()
    logger.info("======= Generation started at: {} ======", now.isoformat())
    logger.info('Fetching logs from Google Sheets ...')
    df_HITL = read_logs()
    # Always make sure it gets created
    HITL_NOTES_DIR.mkdir(exist_ok=True)
    logger.info('Writing csv files for HITL notes ...')
    generate_tables(df_HITL)
    if s3_sync:
        sync_s3(s3_bucket=f"s3://{bucket}")
    end = datetime.utcnow()
    logger.info(
        "======= Generation finished at: {}. Time elapsed ({}) ======",
        end.isoformat(),
        (end - now),
    )


if __name__ == "__main__":
    main()
