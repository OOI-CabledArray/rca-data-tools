"""plots.py

This module contains code for plot creations from various instruments.

"""
from ast import literal_eval
from asyncio.log import logger
from datetime import datetime
from dateutil import parser
import gc
import os
import fsspec
import time
import pandas as pd
from typing import List
import xarray as xr
import matplotlib.pyplot as plt
import importlib

from rca_data_tools.qaqc import dashboard, decimate 
from rca_data_tools.qaqc.qartod import loadQARTOD
from rca_data_tools.qaqc.utils import select_logger, coerce_qartod_executed_to_int

from rca_data_tools.qaqc.constants import (
    SPAN_DICT,
    S3_BUCKET,
    VARIABLE_DICT,
    VARIABLE_PARAM_DICT,
    MULTI_PARAMETER_DICT,
    LOCAL_RANGE_DICT,
    DEPLOYED_RANGE_DICT,
    CALCULATE_DICT,
    CALCULATE_CALLS_DICT,
    FUNCTION_REGISTRY,
    PLOT_DIR_STR,
    PLOT_DIR,
)

# load status dictionary
statusDict = dashboard.loadStatus()

def extractMulti(ds, inst, multi_dict, fileParams):
    multiParam = multi_dict[inst]["parameter"]
    subParams = multi_dict[inst]["subParameters"].strip('"').split(",")
    for i in range(0, len(subParams)):
        newParam = multiParam + "_" + subParams[i]
        ds[newParam] = ds[multiParam][:, i]
        fileParams.append(newParam)
    return ds, fileParams

def run_calculations_for_site(site, siteData):
    """
    Run all configured calculations for a site and append outputs to siteData.
    Requires globals:
        CALCULATE_DICT,
        CALCULATE_CALLS_DICT,
        FUNCTION_REGISTRY
    """

    fileParams = []

    if site not in CALCULATE_DICT:
        return siteData, fileParams

    logger.info(f"Calculating suplimentary data arrays for {site}.")
    for calc_name in CALCULATE_DICT[site]:
        meta = CALCULATE_CALLS_DICT[calc_name]
        func = FUNCTION_REGISTRY[meta["function_key"]]

        # --- gather inputs ---
        args = []
        for name in meta["inputs"]:
            if name == "DATASET":
                args.append(siteData)
            elif name == "site":
                args.append(site)
            else:
                args.append(siteData[name])

        kwargs = meta.get("kwargs", {})

        # --- run calculation ---
        result = func(*args, **kwargs)

        # --- handle outputs ---
        outputs = meta["outputs"]
        results = (result,) if len(outputs) == 1 else result

        for out_name, value in zip(outputs, results):
            if not isinstance(value, xr.DataArray):
                value = xr.DataArray(
                    value,
                    dims=siteData["time"].dims,
                    coords=siteData["time"].coords
                )
            siteData[out_name] = value
            fileParams.append(out_name)

    return siteData, fileParams


def run_dashboard_creation(
    site: str,
    paramList: List[str],
    timeRef: str,
    plotInstrument: str,
    span: str,
    decimationThreshold: int,
    stageDict: dict,
    homebrew_qartod: bool,
    express: bool,
):
    logger = select_logger()
    plt.switch_backend("Agg")  # run locally without changing anything else?

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
    siteData = dashboard.loadData(site, stageDict)
    siteData = coerce_qartod_executed_to_int(siteData)

    fileParams = stageDict[site]["dataParameters"].strip('"').split(",")
    allVar = list(siteData.keys())
    # add qartod and qc flags to fileParams list
    qcStrings = ["_qartod_", "_qc_"]
    qcParams = [
        var
        for var in allVar
        if any(sub in var for sub in fileParams)
        if any(qc in var for qc in qcStrings)
    ]
    fileParams = fileParams + qcParams
    # drop un-used variables from dataset
    dropList = [item for item in allVar if item not in fileParams]
    siteData = siteData.drop(dropList)
    # some instruments have inactive bins for current deployment - ie ADCPs
    if site in DEPLOYED_RANGE_DICT.keys():
        sliceCoord = DEPLOYED_RANGE_DICT[site]["sliceCoord"]
        siteData = siteData.sel(
            {
                sliceCoord: slice(
                    DEPLOYED_RANGE_DICT[site]["lowerB"], DEPLOYED_RANGE_DICT[site]["upperB"]
                )
            }
        )
        logger.warning(
            f"Not all {sliceCoord} coords of {site} "
            "are active on current deployment. Dataset is being subset to active coords."
        )

    logger.debug(f"site array: {siteData}")
    # extract parameters from multi-dimensional array
    if plotInstrument in MULTI_PARAMETER_DICT.keys():
        siteData, fileParams = extractMulti(
            siteData, plotInstrument, MULTI_PARAMETER_DICT, fileParams
        )

    if (
        stageDict[site]["decimationAlgo"] == "lttb"
    ):  # lttb is prefered to preserve points of interest
        if int(span) == 365:
            if len(siteData["time"]) > decimationThreshold:
                # decimate data
                siteData_df = decimate.downsample(siteData, decimationThreshold, logger=logger)
                # turn dataframe into dataset
                del siteData
                gc.collect()
                siteData = xr.Dataset.from_dataframe(siteData_df, sparse=False)
                siteData = siteData.swap_dims({"index": "time"})
                siteData = siteData.reset_coords()

    elif stageDict[site]["decimationAlgo"] == "coarsen":  # use a cruder method for ADCP etc
        if int(span) in [30, 365]:
            if len(siteData["time"]) > decimationThreshold:
                window = int(len(siteData["time"]) / decimationThreshold)
                logger.info(
                    f"{site} unable to be decimated using LTTB. Using xr.coarsen instead"
                )
                siteData = siteData.coarsen(time=window, boundary="trim").mean()
                logger.info(f"Succesfully coarsened time with window of *{window}*.")
                
    if site in CALCULATE_DICT:
        logger.info(f"calculating parameters for {site}...")
        
        siteData, calc_fileParams = run_calculations_for_site(site, siteData)
        fileParams.extend(calc_fileParams)    

    for param in paramList:
        logger.info(f"parameter: {param}")
        variableParams = VARIABLE_DICT[param].strip('"').split(",")
        parameterList = [value for value in variableParams if value in fileParams]
        if len(parameterList) == 0:
            logger.warning(f"Error retriving parameter: {param} from the xarray...")
        else:
            for Yparam in parameterList:  # the parameters actually in xarray (in most cases 1)
                # Yparam = parameterList[0]
                # set up plotting parameters
                if len(parameterList) > 1:
                    imageName_base = PLOT_DIR_STR + site + "_" + Yparam
                    plotTitle = site + " " + Yparam
                else:
                    imageName_base = PLOT_DIR_STR + site + "_" + param
                    plotTitle = site + " " + param
                logger.info(imageName_base)
                paramMin = float(VARIABLE_PARAM_DICT[param]["min"])
                paramMax = float(VARIABLE_PARAM_DICT[param]["max"])
                profile_paramMin = float(VARIABLE_PARAM_DICT[param]["profileMin"])
                profile_paramMax = float(VARIABLE_PARAM_DICT[param]["profileMax"])
                # default local range to standard range if not defined
                paramMin_local = paramMin
                paramMax_local = paramMax
                profile_paramMin_local = profile_paramMin
                profile_paramMax_local = profile_paramMax
                localRange = LOCAL_RANGE_DICT[site][param]

                if "local" in localRange:
                    paramMin_local = localRange["local"][0]
                    paramMax_local = localRange["local"][1]
                if "local_profile" in localRange:
                    profile_paramMin_local = localRange["local_profile"][0]
                    profile_paramMax_local = localRange["local_profile"][1]

                yLabel = VARIABLE_PARAM_DICT[param]["label"]

                # Load overlayData
                overlayData_clim = {}
                overlayData_grossRange = {}
                sensorType = site.split("-")[3][0:5].lower()
                (overlayData_clim, overlayData_grossRange) = loadQARTOD(
                    site, Yparam, sensorType, logger=logger
                )
                overlayData_near = {}
                # overlayData_near = loadNear(site)

                overlayData_anno = {}
                overlayData_anno = dashboard.loadAnnotations(site)

                if "PROFILER" in plotInstrument:
                    profileList = dashboard.loadProfiles(site)
                    pressureParams = VARIABLE_DICT["pressure"].strip('"').split(",")
                    pressureParamList = [
                        value for value in pressureParams if value in fileParams
                    ]
                    if len(pressureParamList) != 1:
                        logger.info("Error retriving pressure parameter!")
                    else:
                        pressParam = pressureParamList[0]
                        paramData = siteData[[Yparam, pressParam]].chunk("auto")
                        flagParams = [item for item in qcParams if Yparam in item]
                        flagParams.extend((Yparam, pressParam))
                        overlayData_flag = siteData[flagParams].chunk("auto")
                        colorMap = "cmo." + VARIABLE_PARAM_DICT[param]["colorMap"]
                        depthMinMax = stageDict[site]["depthMinMax"].strip('"').split(",")
                        if "None" not in depthMinMax:
                            yMin = int(depthMinMax[0])
                            yMax = int(depthMinMax[1])
                        plots = dashboard.plotProfilesGrid(
                            Yparam,
                            pressParam,
                            param,  # short parameter name - see variableMap.csv
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
                        if "ADCP" not in plotInstrument:
                            plots = dashboard.plotProfilesScatter(
                                Yparam,
                                param,
                                pressParam,
                                paramData,
                                plotTitle,
                                timeRef,
                                yMin,
                                yMax,
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
                                homebrew_qartod,
                                express,
                            )
                            plotList.append(plots)
                            depths = stageDict[site]["depths"].strip('"').split(",")
                            if "Single" not in depths:
                                for profileDepth in depths:
                                    paramData_depth = paramData[Yparam].where(
                                        (int(profileDepth) < paramData[pressParam])
                                        & (paramData[pressParam] < (int(profileDepth) + 0.5))
                                    )
                                    overlayData_flag_extract = overlayData_flag.where(
                                        (int(profileDepth) < overlayData_flag[pressParam])
                                        & (
                                            overlayData_flag[pressParam]
                                            < (int(profileDepth) + 0.5)
                                        )
                                    )
                                    plotTitle_depth = (
                                        plotTitle + ": " + profileDepth + " meters"
                                    )
                                    imageName_base_depth = (
                                        imageName_base + "_" + profileDepth + "meters"
                                    )
                                    if overlayData_clim:
                                        overlayData_clim_extract = dashboard.extractClim(
                                            timeRef, profileDepth, overlayData_clim
                                        )
                                    else:
                                        overlayData_clim_extract = pd.DataFrame()
                                    plots = dashboard.plotScatter(
                                        Yparam,
                                        param,
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
                                        "medium",
                                        span,
                                        spanString,
                                        statusDict,
                                        site,
                                        homebrew_qartod,
                                    )
                                    plotList.append(plots)
                else:
                    paramData = siteData[Yparam]
                    flagParams = [item for item in qcParams if Yparam in item]
                    flagParams.append(Yparam)
                    overlayData_flag = siteData[flagParams].chunk("auto")

                    if overlayData_clim:
                        overlayData_clim_extract = dashboard.extractClim(
                            timeRef, "0", overlayData_clim
                        )
                    else:
                        overlayData_clim_extract = pd.DataFrame()
                    # PLOT
                    plots = dashboard.plotScatter(
                        Yparam,
                        param,
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
                        "small",
                        span,
                        spanString,
                        statusDict,
                        site,
                        homebrew_qartod,
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


def organize_images(sync_to_s3=False, bucket_name=S3_BUCKET, fs_kwargs={}):
    for i in PLOT_DIR.iterdir():
        if i.is_file():
            if i.suffix == ".png" or i.suffix == ".svg":
                fname = i.name
                subsite = fname.split("-")[0]

                subsite_dir = PLOT_DIR / subsite
                subsite_dir.mkdir(exist_ok=True)

                destination = subsite_dir / fname
                i.replace(destination)

                # Sync to s3
                if sync_to_s3 is True:
                    import fsspec

                    S3FS = fsspec.filesystem("s3", **fs_kwargs)

                    fs_path = "/".join([bucket_name, PLOT_DIR.name, subsite_dir.name, fname])
                    if S3FS.exists(fs_path):
                        S3FS.rm(fs_path)
                    S3FS.put(str(destination.absolute()), fs_path)
            else:
                print(f"{i} is not a `png` or `svg` file ... will not sort ...")
        else:
            print(f"{i} is not a file ... will not sort ...")


def delete_outdated_images(
    plot_list: List,
    site: str,  # instrument
    span_string: str,
    sync_to_s3: bool,
    bucket_name: str,
    s3fs: fsspec.filesystem,
) -> None:
    # NOTE this may not be working when annotation files are involved - are they piped to plot_list?
    logger = select_logger()

    if sync_to_s3:
        flat_plot_list = [item for sublist in plot_list for item in sublist]
        site_prefix = site.split("-")[0]

        logger.info("Collecting existing 'profile' image files.")

        existing_instrument_files = s3fs.glob(
            f"{bucket_name}/QAQC_plots/{site_prefix}/{site}*"
        )

        existing_profile_files = [
            f for f in existing_instrument_files if "profile" in f and span_string in f
        ]
        new_profile_files = [f for f in flat_plot_list if "profile" in f and span_string in f]

        # Extract only the files name in both types of paths in both lists
        just_file_existing = [f.split("/")[3] for f in existing_profile_files]
        just_file_new = [f.split("/")[1] for f in new_profile_files]
        logger.info(
            f"Number of existing files: {len(just_file_existing)}| Number of new files: {len(just_file_new)}"
        )

        files_to_delete = list(set(just_file_existing) - set(just_file_new))
        files_to_delete_full_path = [
            f"{bucket_name}/QAQC_plots/{site_prefix}/{f}" for f in files_to_delete
        ]

        for f in files_to_delete_full_path:
            s3fs.rm(f)

        logger.info(f"{len(files_to_delete_full_path)} outdated profile images deleted.")

    else:
        logger.info("No s3 sync - no outdated images to delete.")


def delete_outdated_annotations(
    site: str,
    span_string: str,
    sync_to_s3: bool,
    bucket_name: str,
    s3fs: fsspec.filesystem,
) -> None:

    logger = select_logger()
    time.sleep(1) # allow s3 time to reflect newly updated files

    if sync_to_s3:
        site_prefix = site.split("-")[0]
        existing_instrument_files = s3fs.glob(
            f"{bucket_name}/QAQC_plots/{site_prefix}/{site}*"
        )

        existing_anno_files = [
            f for f in existing_instrument_files if "anno" in f and span_string in f
        ]
        anno_svgs = [os.path.splitext(f)[0] for f in existing_anno_files if "svg" in f]
        anno_pngs = [os.path.splitext(f)[0] for f in existing_anno_files if "png" in f]

        anno_files_modified_dict = {}  # k=file : v=when it was modified

        for fpath in existing_anno_files:
            last_modified = s3fs.info(fpath)["LastModified"]
            anno_files_modified_dict[fpath] = last_modified

        overlapping_files = list(set(anno_svgs) & set(anno_pngs))

        outdated_annos_to_delete = []
        for f in overlapping_files:
            svg_time_created = anno_files_modified_dict[f"{f}.svg"]
            png_time_created = anno_files_modified_dict[f"{f}.png"]

            if svg_time_created < png_time_created:
                outdated_annos_to_delete.append(f"{f}.svg")
            elif png_time_created < svg_time_created:
                outdated_annos_to_delete.append(f"{f}.png")

        for f in outdated_annos_to_delete:
            s3fs.rm(f)

        logger.info(f"These outdated files were deleted from s3: {outdated_annos_to_delete}")
    else:
        logger.info("No s3 sync - no outdated images to delete.")
