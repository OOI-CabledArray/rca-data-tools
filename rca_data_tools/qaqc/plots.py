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
import time
import pandas as pd
from typing import List
import xarray as xr
import matplotlib.pyplot as plt

from rca_data_tools.qaqc import dashboard
from rca_data_tools.qaqc import decimate
from rca_data_tools.qaqc import discrete
from rca_data_tools.qaqc import calculate
from rca_data_tools.qaqc.utils import select_logger, coerce_qartod_executed_to_int

from rca_data_tools.qaqc.constants import (
    SPAN_DICT,
    S3_BUCKET,
    variable_dict,
    variable_param_dict,
    multi_parameter_dict,
    local_range_dict,
    deployed_range_dict,
    calculate_dict,
    calculate_strings_dict,
    discrete_sample_dict,
    plot_dir,
    PLOT_DIR,
)

# load status dictionary
status_dict = dashboard.loadStatus()


def extractMulti(ds, inst, multi_dict, file_params):
    multi_param = multi_dict[inst]["parameter"]
    sub_params = multi_dict[inst]["subParameters"].strip('"').split(",")
    for i in range(0, len(sub_params)):
        new_param = multi_param + "_" + sub_params[i]
        ds[new_param] = ds[multi_param][:, i]
        file_params.append(new_param)
    return ds, file_params


def map_concurrency(func, iterator, func_args=(), func_kwargs={}, max_workers=10):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {
            executor.submit(func, i, *func_args, **func_kwargs): i for i in iterator
        }
        for future in concurrent.futures.as_completed(future_to_url):
            data = future.result()
            results.append(data)
    return results


def run_dashboard_creation(
    site,
    param_list,
    time_ref,
    plot_instrument,
    span,
    decimation_threshold,
    stage_dict,
):
    logger = select_logger()
    plt.switch_backend("Agg")  # run locally without changing anything else?

    if isinstance(time_ref, str):
        time_ref = parser.parse(time_ref)

    # Ensure that plot dir is created!
    PLOT_DIR.mkdir(exist_ok=True)

    now = datetime.utcnow()
    plot_list = []
    logger.info(f"site: {site}")
    logger.info(f"span: {span}")

    span_string = SPAN_DICT[span]
    # load data for site
    site_data = dashboard.loadData(site, stage_dict)
    site_data = coerce_qartod_executed_to_int(site_data)

    file_params = stage_dict[site]["dataParameters"].strip('"').split(",")
    all_var = list(site_data.keys())
    # add qartod and qc flags to file_params list
    qc_strings = ["_qartod_", "_qc_"]
    qc_params = [
        var
        for var in all_var
        if any(sub in var for sub in file_params)
        if any(qc in var for qc in qc_strings)
    ]
    file_params = file_params + qc_params
    # drop un-used variables from dataset
    drop_list = [item for item in all_var if item not in file_params]
    site_data = site_data.drop(drop_list)
    # some instruments have inactive bins for current deployment - ie ADCPs
    if site in deployedRange_dict.keys():
        slice_coord = deployedRange_dict[site]["slice_coord"]
        site_data = site_data.sel(
            {
                slice_coord: slice(
                    deployedRange_dict[site]["lowerB"], deployedRange_dict[site]["upperB"]
                )
            }
        )
        logger.warning(
            f"Not all {slice_coord} coords of {site} "
            "are active on current deployment. Dataset is being subset to active coords."
        )

    logger.info(f"site date array: {site_data}")
    # extract parameters from multi-dimensional array
    if plot_instrument in multiParameter_dict.keys():
        site_data, file_params = extractMulti(
            site_data, plot_instrument, multiParameter_dict, file_params
        )

    if (
        stage_dict[site]["decimationAlgo"] == "lttb"
    ):  # lttb is prefered to preserve points of interest
        if int(span) == 365:
            if len(site_data["time"]) > decimation_threshold:
                # decimate data
                siteData_df = decimate.downsample(site_data, decimation_threshold, logger=logger)
                # turn dataframe into dataset
                del site_data
                gc.collect()
                site_data = xr.Dataset.from_dataframe(siteData_df, sparse=False)
                site_data = site_data.swap_dims({"index": "time"})
                site_data = site_data.reset_coords()

    elif stage_dict[site]["decimationAlgo"] == "coarsen":  # use a cruder method for ADCP etc
        if int(span) in [30, 365]:
            if len(site_data["time"]) > decimation_threshold:
                window = int(len(site_data["time"]) / decimation_threshold)
                logger.info(
                    f"{site} unable to be decimated using LTTB. Using xr.coarsen instead"
                )
                site_data = site_data.coarsen(time=window, boundary="trim").mean()
                logger.info(f"Succesfully coarsened time with window of *{window}*.")
    # perform calculations for auxiliary parameters
    if site in calculate_dict:
        for calc in calculate_dict[site]['calculations'].strip('"').split(","):
            if calc in calculateStrings_dict:
                # perform calculation by evaluating string
                exec(calculateStrings_dict[calc]['string'])
                # add new parameter to file_params list
                return_params = calculateStrings_dict[calc]['returnParam'].strip('"').split(",")
                for item in return_params:
                    file_params.append(item)
            else:
                logger.info(f"error calculating parameters: {calc}")

    for param in param_list:
        logger.info(f"parameter: {param}")
        variable_params = variable_dict[param].strip('"').split(",")
        parameter_list = [value for value in variable_params if value in file_params]
        if len(parameter_list) == 0:
            logger.warning(f"Error retriving parameter: {param} from the xarray...")
        else:
            for Yparam in parameter_list:  # the parameters actually in xarray (in most cases 1)
                # Yparam = parameter_list[0]
                # set up plotting parameters
                if len(parameter_list) > 1:
                    imageName_base = plotDir + site + "_" + Yparam
                    plot_title = site + " " + Yparam
                else:
                    imageName_base = plotDir + site + "_" + param
                    plot_title = site + " " + param
                logger.info(imageName_base)
                param_min = float(variable_paramDict[param]["min"])
                param_max = float(variable_paramDict[param]["max"])
                profile_paramMin = float(variable_paramDict[param]["profileMin"])
                profile_paramMax = float(variable_paramDict[param]["profileMax"])
                # default local range to standard range if not defined
                paramMin_local = param_min
                paramMax_local = param_max
                profile_paramMin_local = profile_paramMin
                profile_paramMax_local = profile_paramMax
                local_ranges = str(localRange_dict[site][param])
                if not "nan" in local_ranges:
                    local_range = literal_eval(local_ranges)
                    if "local" in local_range:
                        paramMin_local = local_range["local"][0]
                        paramMax_local = local_range["local"][1]
                    if "local_profile" in local_range:
                        profile_paramMin_local = local_range["local_profile"][0]
                        profile_paramMax_local = local_range["local_profile"][1]

                y_label = variable_paramDict[param]["label"]

                # Load overlayData
                overlayData_clim = {}
                overlayData_grossRange = {}
                sensor_type = site.split("-")[3][0:5].lower()
                (overlayData_grossRange, overlayData_clim) = dashboard.loadQARTOD(
                    site, Yparam, sensor_type, logger=logger
                )
                overlayData_near = {}
                # overlayData_near = loadNear(site)

                overlayData_anno = {}
                overlayData_anno = dashboard.loadAnnotations(site)

                overlayData_disc = pd.DataFrame() #TODO clean up after we descide of final discrete data solution
                # if int(span) == 0:
                #     overlayData_disc = discrete.extractDiscreteOverlay(site,time_ref.year,discreteSample_dict,param)

                if "PROFILER" in plot_instrument:
                    profile_list = dashboard.loadProfiles(site)
                    pressure_params = variable_dict["pressure"].strip('"').split(",")
                    pressure_param_list = [
                        value for value in pressure_params if value in file_params
                    ]
                    if len(pressure_param_list) != 1:
                        logger.info("Error retriving pressure parameter!")
                    else:
                        press_param = pressure_param_list[0]
                        param_data = site_data[[Yparam, press_param]].chunk("auto")
                        flag_params = [item for item in qc_params if Yparam in item]
                        flag_params.extend((Yparam, press_param))
                        overlayData_flag = site_data[flag_params].chunk("auto")
                        color_map = "cmo." + variable_paramDict[param]["color_map"]
                        depth_min_max = stage_dict[site]["depth_min_max"].strip('"').split(",")
                        if "None" not in depth_min_max:
                            y_min = int(depth_min_max[0])
                            y_max = int(depth_min_max[1])
                        plots = dashboard.plotProfilesGrid(
                            Yparam,
                            press_param,
                            param,  # short parameter name - see variableMap.csv
                            param_data,
                            plot_title,
                            y_label,
                            time_ref,
                            y_min,
                            y_max,
                            profile_paramMin,
                            profile_paramMax,
                            profile_paramMin_local,
                            profile_paramMax_local,
                            color_map,
                            imageName_base,
                            overlayData_anno,
                            overlayData_clim,
                            overlayData_near,
                            span,
                            span_string,
                            profile_list,
                            status_dict,
                            site,
                            plot_instrument,
                        )
                        plot_list.append(plots)
                        if "ADCP" not in plot_instrument:
                            plots = dashboard.plotProfilesScatter(
                                Yparam,
                                param,
                                press_param,
                                param_data,
                                plot_title,
                                time_ref,
                                y_min,
                                y_max,
                                profile_paramMin,
                                profile_paramMax,
                                profile_paramMin_local,
                                profile_paramMax_local,
                                imageName_base,
                                overlayData_anno,
                                overlayData_clim,
                                overlayData_disc,
                                overlayData_flag,
                                overlayData_near,
                                span,
                                span_string,
                                profile_list,
                                status_dict,
                                site,
                            )
                            plot_list.append(plots)
                            depths = stage_dict[site]["depths"].strip('"').split(",")
                            if "Single" not in depths:
                                for profile_depth in depths:
                                    paramData_depth = param_data[Yparam].where(
                                        (int(profile_depth) < param_data[press_param])
                                        & (param_data[press_param] < (int(profile_depth) + 0.5))
                                    )
                                    overlayData_flag_extract = overlayData_flag.where(
                                        (int(profile_depth) < overlayData_flag[press_param])
                                        & (
                                            overlayData_flag[press_param]
                                            < (int(profile_depth) + 0.5)
                                        )
                                    )
                                    plotTitle_depth = (
                                        plot_title + ": " + profile_depth + " meters"
                                    )
                                    imageName_base_depth = (
                                        imageName_base + "_" + profile_depth + "meters"
                                    )
                                    if overlayData_clim:
                                        overlayData_clim_extract = dashboard.extractClim(
                                            time_ref, profile_depth, overlayData_clim
                                        )
                                    else:
                                        overlayData_clim_extract = pd.DataFrame()
                                    plots = dashboard.plotScatter(
                                        Yparam,
                                        param,
                                        paramData_depth,
                                        plotTitle_depth,
                                        y_label,
                                        time_ref,
                                        profile_paramMin,
                                        profile_paramMax,
                                        profile_paramMin_local,
                                        profile_paramMax_local,
                                        imageName_base_depth,
                                        overlayData_anno,
                                        overlayData_clim_extract,
                                        overlayData_disc,
                                        overlayData_flag_extract,
                                        overlayData_near,
                                        "medium",
                                        span,
                                        span_string,
                                        status_dict,
                                        site,
                                    )
                                    plot_list.append(plots)
                else:
                    param_data = site_data[Yparam]
                    flag_params = [item for item in qc_params if Yparam in item]
                    flag_params.append(Yparam)
                    overlayData_flag = site_data[flag_params].chunk("auto")

                    if overlayData_clim:
                        overlayData_clim_extract = dashboard.extractClim(
                            time_ref, "0", overlayData_clim
                        )
                    else:
                        overlayData_clim_extract = pd.DataFrame()
                    # PLOT
                    plots = dashboard.plotScatter(
                        Yparam,
                        param,
                        param_data,
                        plot_title,
                        y_label,
                        time_ref,
                        param_min,
                        param_max,
                        paramMin_local,
                        paramMax_local,
                        imageName_base,
                        overlayData_anno,
                        overlayData_clim_extract,
                        overlayData_disc,
                        overlayData_flag,
                        overlayData_near,
                        "small",
                        span,
                        span_string,
                        status_dict,
                        site,
                    )
                    plot_list.append(plots)

                del param_data
                gc.collect()
    del site_data
    gc.collect()
    end = datetime.utcnow()
    elapsed = end - now
    logger.info(f"{site} finished plotting: Time elapsed ({elapsed})")
    return plot_list


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
                print(f"{i} is not a `png` or `svg` file ... skipping ...")
        else:
            print(f"{i} is not a file ... skipping ...")


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
