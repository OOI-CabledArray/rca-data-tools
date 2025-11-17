"""index.py

This module contains code for creating json index files,
for the contents of directories that hold plot pngs and HITL
csv files. These index files are utilized for the QAQC dashboard
frontend application.

"""

import click
import json
import fsspec
from rca_data_tools.qaqc.plots import PLOT_DIR as PLOTSDIR
from rca_data_tools.qaqc.notes import HITL_NOTES_DIR as HITLDIR
from rca_data_tools.qaqc.pipeline import S3_BUCKET

INDEX_FILE = 'index.json'


def create_cloud_index(
    bucket_url,
    storage_options={},
    logger=None,
):
    if logger is None:
        from loguru import logger

    plotsmapper = fsspec.get_mapper(
        '/'.join([bucket_url, PLOTSDIR.name]), **storage_options
    )
    hitlmapper = fsspec.get_mapper(
        '/'.join([bucket_url, HITLDIR.name]), **storage_options
    )

    plots_index = [
        item for item in plotsmapper.keys() if item.endswith(('.png', '.svg'))
    ] # keys of mapper are all file names under root location
    logger.info(f"Current plot index {plots_index}")
    
    hitl_index = [item for item in hitlmapper.keys() if item.endswith('.csv')]

    with plotsmapper.fs.open(
        f"{plotsmapper.root}/{INDEX_FILE}", mode='w'
    ) as f:
        json.dump(plots_index, f)

    with hitlmapper.fs.open(
        f"{hitlmapper.root}/{INDEX_FILE}", mode='w'
    ) as f:
        json.dump(hitl_index, f)


def create_local_index():
    plots_json = PLOTSDIR.joinpath(INDEX_FILE)
    hitl_json = HITLDIR.joinpath(INDEX_FILE)
    plotsmapper = fsspec.get_mapper(str(PLOTSDIR))
    hitlmapper = fsspec.get_mapper(str(HITLDIR))
    plots_index = [
        item for item in plotsmapper.keys() if item.endswith(('.png', 'svg'))
    ]
    hitl_index = [item for item in hitlmapper.keys() if item.endswith('.csv')]

    plots_json.write_text(json.dumps(plots_index))
    hitl_json.write_text(json.dumps(hitl_index))


@click.command()
@click.option('--cloud', is_flag=True, help='Create cloud index if set.')
@click.option('--bucket', default=S3_BUCKET, help='Bucket name for cloud index if not default. (ie staging/test buckets)')
def main(cloud, bucket):
    from loguru import logger

    if cloud:
        create_cloud_index(bucket_url=f"s3://{bucket}", logger=logger)
    else:
        create_local_index()
