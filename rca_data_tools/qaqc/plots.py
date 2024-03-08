# -*- coding: utf-8 -*-
"""plots.py

This module contains code for plot creations from various instruments.

"""
from ast import literal_eval
import concurrent.futures
from datetime import datetime
from dateutil import parser
import gc
import os
import fsspec
import pandas as pd
from typing import List
import xarray as xr
import matplotlib.pyplot as plt

from rca_data_tools.qaqc import dashboard
from rca_data_tools.qaqc import decimate
from rca_data_tools.qaqc.utils import select_logger, coerce_qartod_executed_to_int

from rca_data_tools.qaqc.constants import (
    SPAN_DICT,
    sites_dict,
    variable_dict,
    variable_paramDict,
    multiParameter_dict,
    localRange_dict,
    plotDir, PLOT_DIR
)

# load status dictionary
statusDict = dashboard.loadStatus()


def extractMulti(ds, inst, multi_dict, fileParams):
    multiParam = multi_dict[inst]['parameter']
    subParams = multi_dict[inst]['subParameters'].strip('"').split(',')
    for i in range(0, len(subParams)):
        newParam = multiParam + '_' + subParams[i]
        ds[newParam] = ds[multiParam][:, i]
        fileParams.append(newParam)
    return ds, fileParams


def map_concurrency(
    func, iterator, func_args=(), func_kwargs={}, max_workers=10
):
    results = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {
            executor.submit(func, i, *func_args, **func_kwargs): i
            for i in iterator
        }
        for future in concurrent.futures.as_completed(future_to_url):
            data = future.result()
            results.append(data)
    return results


def run_dashboard_creation(
    site,
    paramList,
    timeRef,
    plotInstrument,
    span,
    decimationThreshold,
):
    logger = select_logger()
    plt.switch_backend('Agg') # run locally without changing anything else?

    if isinstance(timeRef, str):
        timeRef = parser.parse(timeRef)

    # Ensure that plot dir is created!
    PLOT_DIR.mkdir(exist_ok=True)

    now = datetime.utcnow()
    plotList = []
    logger.info(f"site: {site}")
    logger.info(f"span: {span}")

    spanString = SPAN_DICT[span]
    # load data for site
    siteData = dashboard.loadData(site, sites_dict)
    siteData = coerce_qartod_executed_to_int(siteData)

    fileParams = sites_dict[site]['dataParameters'].strip('"').split(',')
    allVar = list(siteData.keys())
    # add qartod and qc flags to fileParams list
    qcStrings = ['_qartod_','_qc_']
    qcParams = [var for var in allVar if any(sub in var for sub in fileParams) if any(qc in var for qc in qcStrings)]
    fileParams = fileParams + qcParams
    # drop un-used variables from dataset
    dropList = [item for item in allVar if item not in fileParams]
    siteData = siteData.drop(dropList)

    logger.info(f"site date array: {siteData}")
    # extract parameters from multi-dimensional array
    if plotInstrument in multiParameter_dict.keys():
        siteData, fileParams = extractMulti(
            siteData, plotInstrument, multiParameter_dict, fileParams
        )

    if int(span) == 365:
        if len(siteData['time']) > decimationThreshold:
            # decimate data
            siteData_df = decimate.downsample(
                siteData, decimationThreshold, logger=logger
            )
            # turn dataframe into dataset
            del siteData
            gc.collect()
            siteData = xr.Dataset.from_dataframe(siteData_df, sparse=False)
            siteData = siteData.swap_dims({'index': 'time'})
            siteData = siteData.reset_coords()

    for param in paramList:
        logger.info(f"parameter: {param}")
        variableParams = variable_dict[param].strip('"').split(',')
        parameterList = [
            value for value in variableParams if value in fileParams
        ]
        if len(parameterList) == 0:
            logger.warning(f"Error retriving parameter: {param} from the xarray...")
        else:
            for Yparam in parameterList: # the parameters actually in xarray (in most cases 1)
                #Yparam = parameterList[0]
                # set up plotting parameters
                if len(parameterList) > 1:
                    imageName_base = plotDir + site + '_' + Yparam
                    plotTitle = site + ' ' + Yparam
                else:
                    imageName_base = plotDir + site + '_' + param
                    plotTitle = site + ' ' + param
                logger.info(imageName_base)
                paramMin = float(variable_paramDict[param]['min'])
                paramMax = float(variable_paramDict[param]['max'])
                profile_paramMin = float(variable_paramDict[param]['profileMin'])
                profile_paramMax = float(variable_paramDict[param]['profileMax'])
                # default local range to standard range if not defined
                paramMin_local = paramMin
                paramMax_local = paramMax
                profile_paramMin_local = profile_paramMin
                profile_paramMax_local = profile_paramMax
                localRanges = str(localRange_dict[site][param])
                if not 'nan' in localRanges:
                    localRange = literal_eval(localRanges)
                    if 'local' in localRange:
                        paramMin_local = localRange['local'][0]
                        paramMax_local = localRange['local'][1]
                    if 'local_profile' in localRange:
                        profile_paramMin_local = localRange['local_profile'][0]
                        profile_paramMax_local = localRange['local_profile'][1]

                yLabel = variable_paramDict[param]['label']

                # Load overlayData
                overlayData_clim = {}
                overlayData_grossRange = {}
                sensorType = site.split('-')[3][0:5].lower()
                (overlayData_grossRange, overlayData_clim) = dashboard.loadQARTOD(
                    site, Yparam, sensorType, logger=logger
                )
                overlayData_near = {}
                # overlayData_near = loadNear(site)

                overlayData_anno = {}
                overlayData_anno = dashboard.loadAnnotations(site)

                if 'PROFILER' in plotInstrument:
                    profileList = dashboard.loadProfiles(site)
                    pressureParams = (
                        variable_dict['pressure'].strip('"').split(',')
                    )
                    pressureParamList = [
                        value for value in pressureParams if value in fileParams
                    ]
                    if len(pressureParamList) != 1:
                        logger.info("Error retriving pressure parameter!")
                    else:
                        pressParam = pressureParamList[0]
                        paramData = siteData[[Yparam, pressParam]].chunk('auto')
                        flagParams = [item for item in qcParams if Yparam in item]
                        flagParams.extend((Yparam, pressParam))
                        overlayData_flag = siteData[flagParams].chunk('auto')
                        colorMap = 'cmo.' + variable_paramDict[param]['colorMap']
                        depthMinMax = (
                            sites_dict[site]['depthMinMax'].strip('"').split(',')
                        )
                        if 'None' not in depthMinMax:
                            yMin = int(depthMinMax[0])
                            yMax = int(depthMinMax[1])
                        plots = dashboard.plotProfilesGrid(
                            Yparam,
                            pressParam,
                            param, # short parameter name - see variableMap.csv
                            paramData,
                            plotTitle,
                            yLabel,
                            timeRef,
                            yMin,
                            yMax,
                            profile_paramMin,
                            profile_paramMax,
                            profile_paramMin_local,
                            profile_paramMax_local,
                            colorMap,
                            imageName_base,
                            overlayData_anno,
                            overlayData_clim,
                            overlayData_near,
                            span,
                            spanString,
                            profileList,
                            statusDict,
                            site,
                            plotInstrument,
                        )
                        plotList.append(plots)
                        if 'ADCP' not in plotInstrument: #TODO try to minimize new if blocks
                            plots = dashboard.plotProfilesScatter(
                                Yparam,
                                pressParam,
                                paramData,
                                plotTitle,
                                timeRef,
                                profile_paramMin,
                                profile_paramMax,
                                profile_paramMin_local,
                                profile_paramMax_local,
                                imageName_base,
                                overlayData_anno,
                                overlayData_clim,
                                overlayData_flag,
                                overlayData_near,
                                span,
                                spanString,
                                profileList,
                                statusDict,
                                site,
                            )
                            plotList.append(plots)
                            depths = sites_dict[site]['depths'].strip('"').split(',')
                            if 'Single' not in depths:
                                for profileDepth in depths:
                                    paramData_depth = paramData[Yparam].where(
                                        (int(profileDepth) < paramData[pressParam])
                                        & (
                                            paramData[pressParam]
                                            < (int(profileDepth) + 0.5)
                                        )
                                    )
                                    overlayData_flag_extract = overlayData_flag.where(
                                        (int(profileDepth) < overlayData_flag[pressParam])
                                        & (
                                            overlayData_flag[pressParam]
                                            < (int(profileDepth) + 0.5)
                                        )
                                    )
                                    plotTitle_depth = (
                                        plotTitle + ': ' + profileDepth + ' meters'
                                    )
                                    imageName_base_depth = (
                                        imageName_base + '_' + profileDepth + 'meters'
                                    )
                                    if overlayData_clim:
                                        overlayData_clim_extract = (
                                            dashboard.extractClim(
                                                timeRef, profileDepth, overlayData_clim
                                            )
                                        )
                                    else:
                                        overlayData_clim_extract = pd.DataFrame()
                                    plots = dashboard.plotScatter(
                                        Yparam,
                                        paramData_depth,
                                        plotTitle_depth,
                                        yLabel,
                                        timeRef,
                                        profile_paramMin,
                                        profile_paramMax,
                                        profile_paramMin_local,
                                        profile_paramMax_local,
                                        imageName_base_depth,
                                        overlayData_anno,
                                        overlayData_clim_extract,
                                        overlayData_flag_extract,
                                        overlayData_near,
                                        'medium',
                                        span,
                                        spanString,
                                        statusDict,
                                        site,
                                    )
                                    plotList.append(plots)
                else:
                    paramData = siteData[Yparam]
                    flagParams = [item for item in qcParams if Yparam in item]
                    flagParams.append(Yparam)
                    overlayData_flag = siteData[flagParams].chunk('auto')

                    if overlayData_clim:
                        overlayData_clim_extract = dashboard.extractClim(
                            timeRef, '0', overlayData_clim
                        )
                    else:
                        overlayData_clim_extract = pd.DataFrame()
                    # PLOT
                    plots = dashboard.plotScatter(
                        Yparam,
                        paramData,
                        plotTitle,
                        yLabel,
                        timeRef,
                        paramMin,
                        paramMax,
                        paramMin_local,
                        paramMax_local,
                        imageName_base,
                        overlayData_anno,
                        overlayData_clim_extract,
                        overlayData_flag,
                        overlayData_near,
                        'small',
                        span,
                        spanString,
                        statusDict,
                        site,
                    )
                    plotList.append(plots)

                del paramData
                gc.collect()
    del siteData
    gc.collect()
    end = datetime.utcnow()
    elapsed = end - now
    logger.info(f"{site} finished plotting: Time elapsed ({elapsed})")
    return plotList


def organize_images(
    sync_to_s3=False, bucket_name='ooi-rca-qaqc-prod', fs_kwargs={}
):
    for i in PLOT_DIR.iterdir():
        if i.is_file():
            if i.suffix == '.png' or i.suffix == '.svg':
                fname = i.name
                subsite = fname.split('-')[0]

                subsite_dir = PLOT_DIR / subsite
                subsite_dir.mkdir(exist_ok=True)

                destination = subsite_dir / fname
                i.replace(destination)

                # Sync to s3
                if sync_to_s3 is True:
                    import fsspec
                    S3FS = fsspec.filesystem('s3', **fs_kwargs)

                    fs_path = '/'.join(
                        [bucket_name, PLOT_DIR.name, subsite_dir.name, fname]
                    )
                    if S3FS.exists(fs_path):
                        S3FS.rm(fs_path)
                    S3FS.put(str(destination.absolute()), fs_path)
            else:
                print(f"{i} is not a `png` or `svg` file ... skipping ...")
        else:
            print(f"{i} is not a file ... skipping ...")


def delete_outdated_images(
    plot_list: List,
    site: str, # instrument
    span_string: str,
    sync_to_s3: bool, 
    bucket_name: str, 
    s3fs: fsspec.filesystem) -> None: 
    # TODO this may not be working when annotation files are involved - are they piped to plot_list?
    logger = select_logger()

    if sync_to_s3:
        flat_plot_list = [item for sublist in plot_list for item in sublist]
        site_prefix = site.split('-')[0]

        logger.info("Collecting existing 'profile' image files.")

        existing_instrument_files = s3fs.glob(f"{bucket_name}/QAQC_plots/{site_prefix}/{site}*")

        existing_profile_files = [f for f in existing_instrument_files if 'profile' in f and span_string in f]
        new_profile_files = [f for f in flat_plot_list if 'profile' in f and span_string in f]

        # Extract only the files name in both types of paths in both lists
        just_file_existing = [f.split("/")[3] for f in existing_profile_files]
        just_file_new = [f.split("/")[1] for f in new_profile_files]
        logger.info(f"Number of existing files: {len(just_file_existing)}| Number of new files: {len(just_file_new)}")

        files_to_delete = list(set(just_file_existing) - set(just_file_new))
        files_to_delete_full_path = [f"{bucket_name}/QAQC_plots/{site_prefix}/{f}" for f in files_to_delete]

        for f in files_to_delete_full_path:
            s3fs.rm(f)

        logger.info(f"{len(files_to_delete_full_path)} outdated profile images deleted.")

    else:
        logger.info("No s3 sync - no outdated images to delete.")


def delete_outdated_annotations(
    site: str, # instrument
    span_string: str,
    sync_to_s3: bool, 
    bucket_name: str, 
    s3fs: fsspec.filesystem) -> None: 

    logger = select_logger()

    if sync_to_s3:
        #TODO could turn this an delete_outdated imgs into a single function.
        site_prefix = site.split('-')[0]
        existing_instrument_files = s3fs.glob(f"{bucket_name}/QAQC_plots/{site_prefix}/{site}*")

        existing_anno_files = [f for f in existing_instrument_files if 'anno' in f and span_string in f]
        anno_svgs = [os.path.splitext(f)[0] for f in existing_anno_files if 'svg' in f]
        anno_pngs = [os.path.splitext(f)[0] for f in existing_anno_files if 'png' in f]

        anno_files_modified_dict = {} # k=file : v=when it was modified

        for fpath in existing_anno_files:
            last_modified = s3fs.info(fpath)['LastModified']
            anno_files_modified_dict[fpath] = last_modified

        overlapping_files = list(set(anno_svgs) & set(anno_pngs))

        outdated_annos_to_delete = []
        for f in overlapping_files:
            svg_time_created = anno_files_modified_dict[f'{f}.svg']
            png_time_created = anno_files_modified_dict[f'{f}.png']

            if svg_time_created < png_time_created:
                outdated_annos_to_delete.append(f'{f}.svg')
            elif png_time_created < svg_time_created:
                outdated_annos_to_delete.append(f'{f}.png')

        for f in outdated_annos_to_delete:
            s3fs.rm(f)

        logger.info(f'These outdated files were deleted from s3: {outdated_annos_to_delete}')
    else:
        logger.info("No s3 sync - no outdated images to delete.")
