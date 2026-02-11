from typing import List
import fsspec
import os

from prefect import task, flow
from prefect.states import Cancelled
from prefect import get_run_logger
from importlib.metadata import distributions

from rca_data_tools.qaqc.plots import (
    organize_images,
    run_dashboard_creation,
    delete_outdated_images,
    delete_outdated_annotations,
)
from rca_data_tools.qaqc.utils import get_s3_kwargs
from rca_data_tools.qaqc.visual_data import cam_qaqc_stacked_bar
from rca_data_tools.qaqc.constants import (
    S3_BUCKET, 
    SPAN_DICT,
    instrument_dict,
    sites_dict,
    stage2_dict,
    stage3_dict,
)


@task
def dashboard_creation_task(
    site, 
    timeString, 
    span, 
    threshold,
    stage,
    homebrew_qartod,
    ):
    """
    Prefect task for running dashboard creation
    """
    stage_map = {1: sites_dict, 2: stage2_dict, 3: stage3_dict}
    stage_dict = stage_map[stage]

    site_ds = stage_dict[site]
    plotInstrument = site_ds['instrument']
    paramList = (
        instrument_dict[plotInstrument]['plotParameters']
        .replace('"', '')
        .split(',')
    )

    plotList = run_dashboard_creation(
        site,
        paramList,
        timeString,
        plotInstrument,
        span,
        threshold,
        stage_dict,
        homebrew_qartod,
    )
    return plotList
        

@task 
def delete_outdated_images_task(
    plotList: List,
    site: str, # instrument
    span: str,
    sync_to_s3: bool, 
    bucket_name: str, 
    s3fs: fsspec.filesystem) -> None:
    """
    Prefect task for deleting outdating image files that would otherwise not be overwritten.
    """
    span_string = SPAN_DICT[span]

    delete_outdated_images(
        plot_list=plotList,
        site=site,
        span_string=span_string,
        sync_to_s3=sync_to_s3,
        bucket_name=bucket_name,
        s3fs=s3fs,
    )

    delete_outdated_annotations(
        site=site,
        span_string=span_string,
        sync_to_s3=sync_to_s3,
        bucket_name=bucket_name,
        s3fs=s3fs,
    )


@task
def organize_images_task(
    plotList=[], fs_kwargs={}, sync_to_s3=False, s3_bucket=S3_BUCKET
):
    """
    Prefect task for organizing the plot pngs to their appropriate directories
    """
    logger = get_run_logger()
    logger.info(f"plot list: {plotList}")
    logger.info(f"sync_to_s3: {sync_to_s3}")
    logger.info(f"s3_bucket: {s3_bucket}")

    if len(plotList) > 0:
        organize_images(
            sync_to_s3=sync_to_s3, fs_kwargs=fs_kwargs, bucket_name=s3_bucket
        )
    else:
        return Cancelled(message="No plots found to be organized.")
    

@flow(timeout_seconds=86400) # 28800 TODO switch back after staging qartod tests
def qaqc_pipeline_flow(
    site: str,
    timeString: str,
    span: str='1',
    threshold: int=5000000,
    stage: int=None,
    homebrew_qartod: bool=False,
    # cloud args
    fs_kwargs: dict={},
    sync_to_s3: bool=True,
    s3_bucket: str=S3_BUCKET,
    ):

    logger = get_run_logger()

    # log python package versions on cloud machine
    installed_packages = {dist.metadata["Name"]: dist.version for dist in distributions()}
    logger.info(f"Installed packages: {installed_packages}")
    logger.info(f"Available cpu cores on runner machine: {os.cpu_count()}")

    if 'CAM' in site:
        logger.warning("Running camera qaqc routine.")
        plotList = cam_qaqc_stacked_bar(
            site=site,
            time_string=timeString,
            span=span,
        )

    else:
    # Run dashboard creation task
        plotList = dashboard_creation_task(
            site=site,
            timeString=timeString,
            span=span,
            threshold=threshold,
            stage=stage,
            homebrew_qartod=homebrew_qartod,
        )
        
    fs_kwargs = get_s3_kwargs()
    S3FS = fsspec.filesystem('s3', **fs_kwargs)
    # Delete outdated profile and annotation images
    delete_outdated_images_task(
        plotList=plotList,
        site=site,
        span=span,
        sync_to_s3=sync_to_s3,
        bucket_name=s3_bucket,
        s3fs=S3FS
    )

    # Run organize images task
    organize_images_task(
        plotList=plotList,
        sync_to_s3=sync_to_s3,
        fs_kwargs=fs_kwargs,
        s3_bucket=s3_bucket,
    )
