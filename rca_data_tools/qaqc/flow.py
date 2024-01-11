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

S3_BUCKET = 'ooi-rca-qaqc-prod'
SPAN_DICT = {'1': 'day', '7': 'week', '30': 'month', '365': 'year'}

@task
def dashboard_creation_task(
    site, 
    timeString, 
    span, 
    threshold, 
    #logger
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
        #logger,
    )
    return plotList
    # except Exception as e:
        # raise prefect_signals.FAIL(
        #     message=f"PNG Creation Failed for {site}: {e}"
        # )
        # return Failed(message=f"PNG Creation Failed for {site}: {e}")
        

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
    logger.info(f"fs_kwargs: {fs_kwargs}")

    if len(plotList) > 0:
        organize_images(
            sync_to_s3=sync_to_s3, fs_kwargs=fs_kwargs, bucket_name=s3_bucket
        )
    else:
        #raise prefect_signals.SKIP(message="No plots found to be organized.")
        return Cancelled(message="No plots found to be organized.")
    

now = datetime.datetime.utcnow()

@flow
def qaqc_pipeline_flow(
    site: str,
    timeString: str=now.strftime('%Y-%m-%d'),
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

    # Run dashboard creation task
    plotList = dashboard_creation_task(
        site=site,
        timeString=timeString,
        span=span,
        threshold=threshold,
    )

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
