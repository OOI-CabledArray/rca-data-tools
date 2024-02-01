from typing import List
import datetime
import pkg_resources

from prefect import task, flow
from prefect.states import Failed, Cancelled
from prefect import get_run_logger

from rca_data_tools.qaqc.plots import (
    instrument_dict,
    organize_images,
    run_dashboard_creation,
    delete_outdated_images,
    sites_dict,
)
from rca_data_tools.qaqc.utils import get_s3_kwargs
from rca_data_tools.qaqc.visual_data import cam_qaqc_stacked_bar
from rca_data_tools.qaqc.constants import S3_BUCKET, SPAN_DICT


@task
def dashboard_creation_task(
    site, 
    timeString, 
    span, 
    threshold, 
    ):
    """
    Prefect task for running dashboard creation
    """
    site_ds = sites_dict[site]
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
    )
    return plotList
        

@task 
def delete_outdated_images_task(
    plotList: List,
    site: str, # instrument
    span: str,
    sync_to_s3: bool, 
    bucket_name: str, 
    fs_kwargs={}) -> None:
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
        fs_kwargs=fs_kwargs,
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
    logger.info(f"s3_bucket: {S3_BUCKET}")

    if len(plotList) > 0:
        organize_images(
            sync_to_s3=sync_to_s3, fs_kwargs=fs_kwargs, bucket_name=s3_bucket
        )
    else:
        return Cancelled(message="No plots found to be organized.")
    

@flow
def qaqc_pipeline_flow(
    site: str,
    timeString: str,
    span: str='1',
    threshold: int=1000000,
    # For organizing pngs
    fs_kwargs: dict={},
    sync_to_s3: bool=True,
    s3_bucket: str=S3_BUCKET,
    ):

    logger = get_run_logger()

    # log python package versions on cloud machine
    installed_packages = {p.project_name: p.version for p in pkg_resources.working_set}
    logger.info(f"Installed packages: {installed_packages}")

    if 'CAMDS' in site:
        logger.warning("Running digital still qaqc routine!")
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
        )

    fs_kwargs = get_s3_kwargs()
    # Delete outdated images
    delete_outdated_images_task(
        plotList=plotList,
        site=site,
        span=span,
        sync_to_s3=sync_to_s3,
        bucket_name=s3_bucket,
        fs_kwargs=fs_kwargs
    )

    # Run organize images task
    organize_images_task(
        plotList=plotList,
        sync_to_s3=sync_to_s3,
        fs_kwargs=fs_kwargs,
        s3_bucket=s3_bucket,
    )
