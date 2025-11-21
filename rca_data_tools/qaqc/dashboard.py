"""dashboard.py

This module contains code for creating pngs to feed into the QAQC dashboard.

"""

import matplotlib

import ast
from datetime import datetime, timedelta
from dateutil import parser
from prefect import task

import gc
import io
import json
import numpy as np
import pandas as pd
import re
import requests
import s3fs
import statistics as st
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean # noqa
from scipy.interpolate import griddata
import textwrap as tw
import xml.etree.ElementTree as et

from rca_data_tools.qaqc.utils import select_logger, save_fig, get_s3_kwargs
from rca_data_tools.qaqc.constants import variable_param_dict, status_colors, discrete_sample_dict
INPUT_BUCKET = "ooi-data/"


def loadAnnotations(site):
    logger = select_logger()
    anno = {}
    fs = s3fs.S3FileSystem(**get_s3_kwargs())
    anno_file = INPUT_BUCKET + 'annotations/' + site + '.json'
    if fs.exists(anno_file):
        anno_store = fs.open(anno_file)
        anno = json.load(anno_store)
    else:
        logger.warning(f"error retrieving annotation history for {site}")

    return anno


def pressureBracket(pressure, clim_dict):
    bracket_list = []
    press_bracket = 'notFound'

    for bracket in clim_dict['1'].keys():
        bracket_list.append(ast.literal_eval(bracket))
    if pressure < bracket_list[0][0]:
        press_bracket = bracket_list[0]
    elif pressure > bracket_list[-1][1] - 1:
        press_bracket = bracket_list[-1]
    else:
        for bracket in bracket_list:
            if (pressure >= bracket[0]) & (pressure < bracket[1]):
                press_bracket = bracket
                break

    return press_bracket



def extractClimProfiles(clim_months, overlayData_clim):
    climatology = {}
    if overlayData_clim:
        for clim_month in clim_months:
            depth = []
            clim_minus_3std = []
            clim_plus_3std = []
            clim_data = []
            for bracket in overlayData_clim[str(clim_month)].keys():
                depth.append(st.mean(ast.literal_eval(bracket)))
                clim0=ast.literal_eval(overlayData_clim[str(clim_month)][bracket])[0]
                clim_minus_3std.append(clim0)
                clim1=ast.literal_eval(overlayData_clim[str(clim_month)][bracket])[1]
                clim_plus_3std.append(clim1)
                clim_data.append(st.mean([clim0,clim1]))
                climatology[str(clim_month)] = {'depth':depth,'clim_minus_3std':clim_minus_3std,'clim_plus_3std':clim_plus_3std,'clim_data':clim_data}

    return climatology



def extractClim(time_ref, profile_depth, overlayData_clim):

    depth = float(profile_depth)
    clim_bracket = pressureBracket(depth, overlayData_clim)
    clim_time = []
    clim_minus_3std = []
    clim_plus_3std = []
    clim_data = []

    if 'notFound' in clim_bracket:
        climInterpolated_hour = pd.DataFrame()
    else:
        for i in range(1, 13):
            clim_month = i
            climatology = ast.literal_eval(
                overlayData_clim[str(clim_month)][str(clim_bracket)]
            )
            # current year
            clim_time.append(datetime(time_ref.year, i, 15))
            clim_minus_3std.append(climatology[0])
            clim_plus_3std.append(climatology[1])
            clim_data.append(st.mean([climatology[0], climatology[1]]))
            # extend climatology to previous year
            clim_time.append(datetime(time_ref.year - 1, i, 15))
            clim_minus_3std.append(climatology[0])
            clim_plus_3std.append(climatology[1])
            clim_data.append(st.mean([climatology[0], climatology[1]]))
            # extend climatology to next year
            clim_time.append(datetime(time_ref.year + 1, i, 15))
            clim_minus_3std.append(climatology[0])
            clim_plus_3std.append(climatology[1])
            clim_data.append(st.mean([climatology[0], climatology[1]]))

        zipped = zip(clim_time, clim_minus_3std, clim_plus_3std, clim_data)
        zipped = list(zipped)
        sort_clim = sorted(zipped, key=lambda x: x[0])

        clim_series = pd.DataFrame(
            sort_clim,
            columns=['clim_time', 'clim_minus_3std', 'clim_plus_3std', 'clim_data'],
        )
        clim_series.set_index(['clim_time'], inplace=True)

        upsampled_hour = clim_series.resample('H')
        climInterpolated_hour = upsampled_hour.interpolate(method='linear')

    return climInterpolated_hour



def gridProfiles(ds, pressure_name, variable_name, profile_indices, profile_depth, profile_depth_grid):
    """
    Interpolate profiles onto a pressure grid for contour plotting.

    Args:
        ds (xarray.Dataset): Input dataset containing profile data.
        pressure_name (str): Name of the pressure variable in `ds`.
        variable_name (str): Name of the data variable in `ds` to interpolate.
        profile_indices (pandas.DataFrame): DataFrame of profile indices 
            (start, peak, end) for each profile.
        profile_depth (float): Maximum depth of profiles at the site.
        profile_depth_grid (float): Grid spacing for the depth axis 
            (e.g., 0.5 m shallow, 5 m deep).
    
    Returns:
        tuple: A tuple containing three elements:
            - grid_x (numpy.ndarray): 1D array of time values the length of the number of profiles.
            - grid_y (numpy.ndarray): 1D array of depth values for the pressure grid.
            - grid_z (numpy.ndarray): 2D array of interpolated variable values on the (depth, time) grid.
    """

    mask = (profile_indices['start'] > ds.time[0].values) & (profile_indices['end'] <= ds.time[-1].values)
    profile_indices = profile_indices.loc[mask]

    if profile_indices.empty:
        grid_x = np.zeros(1)
        grid_y = np.zeros(1)
        grid_z = np.zeros(1)

    else:
        profile_indices = profile_indices.reset_index()

        descent_samples = ['pco2_seawater','ph_seawater']

        if variable_name in descent_samples:
            start = 'peak'
            end = 'end'
            invert = False
        else:
            start = 'start'
            end = 'peak'
            invert = True

        grid_x = np.zeros(len(profile_indices))
        grid_y = np.arange(0, profile_depth, profile_depth_grid)
        grid_z = np.zeros((len(grid_y),len(grid_x)))
        for index, row in profile_indices.iterrows():
            start_time = row[start]
            end_time = row[end]
            ds_sub = ds.sel(time=slice(start_time,end_time))
            if invert:
                variable = np.flip(ds_sub[variable_name].values)
                pressure = np.flip(ds_sub[pressure_name].values)
            else:
                variable = ds_sub[variable_name].values
                pressure = ds_sub[pressure_name].values
            if (len(ds_sub['time']) > 0) and (len(pressure) > 1):
                grid_x[index] = row['peak'].timestamp()
                try:
                    profile = np.interp(grid_y,pressure,variable)
                    grid_z[:,index] = profile
                    min_press = min(pressure)
                    max_press = max(pressure)
                    if min_press > 5:
                        press_mask_min = np.where(grid_y < min_press)
                        grid_z[press_mask_min,index] = np.nan
                    if max_press < 185:
                        press_mask_max = np.where(grid_y > max_press)
                        grid_z[press_mask_max,index] = np.nan 
                except:
                    grid_z[:,index] = np.nan
            else:
                grid_z[:,index] = np.nan

    return(grid_x, grid_y, grid_z)

    
def loadDeploymentHistory(ref_des):
    logger = select_logger()
    deploy_history = {}
    (site, node, sensor1, sensor2) = ref_des.split('-')
    date_columns = ['startDateTime','stopDateTime']
    gh_baseURL = 'https://raw.githubusercontent.com/oceanobservatories/asset-management/master/deployment/'
    deploy_url = gh_baseURL + site + '_Deploy.csv'
    
    download = requests.get(deploy_url)
    if download.status_code == 200:
        df = pd.read_csv(io.StringIO(download.content.decode('utf-8')),parse_dates=date_columns)
        df_sort = df.sort_values(by=["Reference Designator","startDateTime"],ascending=False)
        for i in df_sort['Reference Designator'].unique():
            deploy_history[i] = [{'deployDate':df_sort['startDateTime'][j],'deployEnd':df_sort['stopDateTime'][j],
                                'deployNum':df_sort['deploymentNumber'][j]} 
                                for j in df_sort[df_sort['Reference Designator']==i].index]

        
    else:
        logger.warning(f"error retrieving deployment history for {site}")

    return deploy_history



def loadProfiles(ref_des):
    logger = select_logger()

    profile_list = []
    date_columns = ['start','peak','end']
    if len(ref_des) > 8:
        (site, node, sensor1, sensor2) = ref_des.split('-')
    else:
        site = ref_des
    # URL on the Github where the csv files are stored
    github_url = 'https://github.com/OOI-CabledArray/profile_indices/' 
    gh_baseURL = 'https://raw.githubusercontent.com/OOI-CabledArray/profile_indices/main/'

    page = requests.get(github_url).text
    file_names = list(set(re.findall(site + '_profiles_[0-9]{4}\.csv',page)))

    if file_names:
        profiles_partial = []
        headers = {'User-Agent': 'RCA-profile-fetcher'} # so github doesn't block us?
        logger.info("fetching profiles from github...")
        for file in file_names:
            profiles_URL = gh_baseURL + file
            download = requests.get(profiles_URL, headers=headers)
            if download.status_code == 200:
                data = pd.read_csv(io.StringIO(download.content.decode('utf-8')),parse_dates=date_columns)
                profiles_partial.append(data)

        profile_list = pd.concat(profiles_partial, ignore_index=True)
        profile_list = profile_list.sort_values('start')
    
    return profile_list




def loadQARTOD(ref_des, param, sensor_type, logger=select_logger()):

    rename_map = {
                 'sea_water_temperature':'seawater_temperature',
                 'sea_water_practical_salinity':'practical_salinity',
                 'sea_water_pressure':'seawater_pressure',
                 'sea_water_density':'density',
                 #'ph_seawater':'seawater_ph',
                 }

    if param in rename_map:
        param = rename_map[param]

    (site, node, sensor1, sensor2) = ref_des.split('-')
    sensor = sensor1 + '-' + sensor2

    # if parameter is oxygen sensor in ctd stream, replace sensor type
    if sensor_type == 'ctdpf' and 'oxygen' in param:
        if 'SF' in node:
            sensor_type = 'dofst'
        else:
            sensor_type = 'dosta'

    # Load climatology and gross range values

    github_base_url = 'https://raw.githubusercontent.com/oceanobservatories/qc-lookup/master/qartod/'

    if sensor_type == 'phsen':
        param = 'seawater_ph'

    clim_URL = (
        github_base_url
        + sensor_type
        + '/climatology_tables/'
        + ref_des
        + '-'
        + param
        + '.csv'
    )
    grossRange_URL = (
        github_base_url
        + sensor_type
        + '/'
        + sensor_type
        + '_qartod_gross_range_test_values.csv'
    )

    download = requests.get(grossRange_URL)
    if download.status_code == 200:
        df_grossRange = pd.read_csv(
            io.StringIO(download.content.decode('utf-8'))
        )
        qc_config = df_grossRange.qc_config[
            (df_grossRange.subsite == site)
            & (df_grossRange.node == node)
            & (df_grossRange.sensor == sensor)
            & (df_grossRange.parameters.str.contains(param))
        ]
        if len(qc_config) > 0:
            qcConfig_json = qc_config.values[0].replace("'", "\"")
            grossRange_dict = json.loads(qcConfig_json)
        else:
            logger.warning(
                f"error retrieving gross range data for {ref_des} {param} {sensor_type}"
            )
            grossRange_dict = {}
    else:
        logger.warning(
            f"error retrieving gross range data for {ref_des} {param} {sensor_type}"
        )
        grossRange_dict = {}

    download = requests.get(clim_URL)
    if download.status_code == 200:
        df_clim = pd.read_csv(io.StringIO(download.content.decode('utf-8')))
        clim_rename = {
            'Unnamed: 0': 'depth',
            '[1, 1]': '1',
            '[2, 2]': '2',
            '[3, 3]': '3',
            '[4, 4]': '4',
            '[5, 5]': '5',
            '[6, 6]': '6',
            '[7, 7]': '7',
            '[8, 8]': '8',
            '[9, 9]': '9',
            '[10, 10]': '10',
            '[11, 11]': '11',
            '[12, 12]': '12',
        }

        df_clim.rename(columns=clim_rename, inplace=True)
        clim_dict = df_clim.set_index('depth').to_dict()
    else:
        logger.warning(
            f"error retrieving climatology data for {ref_des} {param} {sensor_type}"
        )
        clim_dict = {}

    return (grossRange_dict, clim_dict)

def loadStatus():
    status_response = requests.get("https://nereus.ooirsn.uw.edu/api/public/v1/instruments/operational-status").text
    status = json.loads(status_response)

    return status


def loadData(site, sites_dict):
    fs = s3fs.S3FileSystem(**get_s3_kwargs())
    zarr_dir = INPUT_BUCKET + sites_dict[site]['zarrFile']
    zarr_store = fs.get_mapper(zarr_dir)
    # NOTE: in future only request parameters listed in sites_dict[site][dataParameters]?
    # requestParams = sites_dict[site]['dataParameters'].strip('"').split(',')
    ds = xr.open_zarr(zarr_store, consolidated=True)

    return ds

def listDeployTimes(deploy_dict):
    deploy_times = []
    for deploy in deploy_dict:
        deploy_times.append(deploy['deployDate'])
        
    return deploy_times



def annoInRange(start_date,end_date,anno_start,anno_end):
    if (anno_start >= end_date) or (anno_end is not None and anno_end <= start_date):
        in_range = False
        start_anno_line = None
        end_anno_line = None
    else:
        in_range = True
        start_anno_line = anno_start
        end_anno_line = anno_end
        if anno_start < start_date:
            start_anno_line = start_date
        if anno_end is None or (anno_end is not None and anno_end > end_date):
            end_anno_line = end_date

    return in_range,start_anno_line,end_anno_line



def annoXnormalize(start_date,end_date,anno_min_date,anno_max_date):
    anno_xmin = (anno_min_date - start_date) / (end_date - start_date)
    anno_xmax =  (anno_max_date - start_date) / (end_date - start_date)

    return anno_xmin,anno_xmax



def saveAnnos_SVG(anno_lines,file_object,file_name):
    et.register_namespace("", "http://www.w3.org/2000/svg")
    # Create XML tree from the SVG file.
    tree, xmlid = et.XMLID(file_object.getvalue())
    tree.set('onload', 'init(event)')

    for i in anno_lines:
        # Get the index of the shape
        index = anno_lines.index(i)
        # Hide the tooltips
        tooltip = xmlid[f'label_{index}']
        tooltip.set('visibility', 'hidden')
        # Assign onmouseover and onmouseout callbacks to patches.
        mypatch = xmlid[f'anno_{index}']
        mypatch.set('onmouseover', "ShowTooltip(this)")
        mypatch.set('onmouseout', "HideTooltip(this)")

    # This is the script defining the ShowTooltip and HideTooltip functions.
    script = """
        <script type="text/ecmascript">
        <![CDATA[

        function init(event) {
            if ( window.svgDocument == null ) {
                svgDocument = event.target.ownerDocument;
                }
            }

        function ShowTooltip(obj) {
            var cur = obj.id.split("_")[1];
            var tip = svgDocument.getElementById('label_' + cur);
            tip.setAttribute('visibility', "visible")
            }

        function HideTooltip(obj) {
            var cur = obj.id.split("_")[1];
            var tip = svgDocument.getElementById('label_' + cur);
            tip.setAttribute('visibility', "hidden")
            }

        ]]>
        </script>
         """

    # Insert the script at the top of the file and save it.
    tree.insert(0, et.XML(script))
    et.ElementTree(tree).write(file_name + '.svg')

@task
def plotProfilesGrid(
    Yparam, # variable of interest
    press_param, # pressure parameter
    param_nickname, # short parameter name - see variableMap.csv
    param_data, # xr.data_array
    plot_title,
    z_label,
    time_ref,
    y_min,
    y_max,
    z_min,
    z_max,
    zMin_local,
    zMax_local,
    color_map,
    fileName_base,
    overlayData_anno,
    overlayData_clim,
    overlayData_near,
    span,
    span_string,
    profile_list,
    status_dict,
    site,
    plot_instrument,
):
    logger = select_logger()

    ### QC check for grid...this will be replaced with a new range for "gross range"
    if 'pco2' in Yparam:
        param_data = param_data.where((param_data[Yparam] < 2000).compute(), drop=True)
    #if 'par' in Yparam:
    #    param_data = param_data.where((param_data[Yparam] > 0) & (param_data[Yparam] < 2000), drop=True)

    # Initiate file_name list
    file_name_list = []
    dpi = 300
    # Plot Overlays
    overlays = ['clim', 'anno', 'none']

    # Data Ranges
    ranges = ['full', 'standard', 'local']

    balance_big = plt.get_cmap('cmo.balance', 512)
    balance_blue = ListedColormap(balance_big(np.linspace(0, 0.5, 256)))

    unix_epoch = np.datetime64(0, 's')
    one_second = np.timedelta64(1, 's')

    status_string = status_dict[site]


    def plotter(Xx, Yy, Zz, plot_type, color_bar, annotation, params, press_label, plot_function=None):

        logger.info(f"params:{params}")
        logger.info(f"plot-type: {plot_type}")
        plt.close('all')
        plt.rcParams["font.family"] = "serif"

        fig, ax = plt.subplots()
        fig.set_size_inches(5, 1.75)
        fig.patch.set_facecolor('white')
        plt.title(plot_title, fontsize=4, loc='left')
        plt.title(status_string, fontsize=4, fontweight=0, color=status_colors[status_string], loc='right', style='italic' )
        plt.ylabel(press_label, fontsize=4)
        ax.tick_params(direction='out', length=2, width=0.5, labelsize=4)
        ax.ticklabel_format(useOffset=False)
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        formatter.formats = [
            '%y',  # ticks are mostly years
            '%b',  # ticks are mostly months
            '%m/%d',  # ticks are mostly days
            '%H h',  # hrs
            '%H:%M',  # min
            '%S.%f',
        ]  # secs
        formatter.zero_formats = [
            '',  # ticks are mostly years, no need for zero_format
            '%b-%Y',  # ticks are mostly months, mark month/year
            '%m/%d',  # ticks are mostly days, mark month/year
            '%m/%d',  # ticks are mostly hours, mark month and day
            '%H',  # ticks are montly mins, mark hour
            '%M',
        ]  # ticks are mostly seconds, mark minute

        formatter.offset_formats = [
            '',
            '',
            '',
            '',
            '',
            '',
        ]

        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(False)
        ax.invert_yaxis()
        plt.xlim(x_min, x_max)
        
        if 'contour' in plot_type:
            if 'full' in params['range']:
                if plot_function == 'meshgrid':
                    graph = ax.pcolormesh(Xx, Yy, Zz, cmap=color_bar, rasterized=True)
                else:
                    graph = ax.contourf(Xx, Yy, Zz, 50, cmap=color_bar, rasterized=True, linewidths=0)
            else:
                color_range = params['vmax'] - params['vmin']
                cbarticks = np.arange(params['vmin'], params['vmax'], color_range / 50)
                if plot_function == 'meshgrid':
                    graph = ax.pcolormesh(Xx, Yy, Zz, vmax=params['vmax'], vmin=params['vmin'], cmap=color_bar, rasterized=True)
                else:
                    graph = ax.contourf(Xx, Yy, Zz, cbarticks, cmap=color_bar, rasterized=True, linewidths=0)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            cbar = plt.colorbar(graph, cax=cax)
            if 'standard' in params['range']:
                graph.set_clim(params['vmin'], params['vmax'])
            cbar.update_ticks()
            cbar.formatter.set_useOffset(False)
            cbar.ax.set_ylabel(z_label, fontsize=4)
            cbar.ax.tick_params(length=2, width=0.5, labelsize=4)
            cbar.solids.set_edgecolor("face")
        
        if 'empty' in plot_type:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            for axis in ['top','bottom','left','right']:
                cax.spines[axis].set_linewidth(0)
            cax.set_xticks([])
            cax.set_yticks([])
            plt.annotate(annotation, xy=(0.3, 0.5), xycoords='figure fraction')

        if 'clim' in plot_type:
            color_range = params['vmax'] - params['vmin']
            cbarticks = np.arange(params['vmin'],params['vmax'],color_range/50)
            if 'yes' in params['norm']:
                divnorm = colors.TwoSlopeNorm(
                    vmin=params['vmin'], vcenter=0,vmax=params['vmax']
                    )
                graph = ax.contourf(Xx, Yy, Zz, cbarticks, cmap=color_bar,vmin=params['vmin'],
                                vmax=params['vmax'], norm=divnorm, rasterized=True)
            else:
                graph = ax.contourf(Xx, Yy, Zz, cbarticks, cmap=color_bar,vmin=params['vmin'],
                                vmax=params['vmax'], rasterized=True)
            m = ScalarMappable(cmap=graph.get_cmap())
            m.set_array(graph.get_array())
            m.set_clim(graph.get_clim())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            cbar = plt.colorbar(graph, cax=cax)
            cbar.update_ticks()
            cbar.formatter.set_useOffset(False)
            cbar.ax.set_ylabel(z_label, fontsize=4)
            cbar.ax.tick_params(length=2, width=0.5, labelsize=4)
        return (fig, ax)

    logger.info(f'plotting grid for time_span: {span} ')

    if 'deploy' in span_string:
        deploy_history = loadDeploymentHistory(site)
        deploy_times = listDeployTimes(deploy_history[site])

        timeRef_deploy = deploy_times[0]
        start_date = timeRef_deploy - timedelta(days=15)
        end_date = timeRef_deploy + timedelta(days=15)
        x_min = start_date - timedelta(days=15 * 0.002)
        x_max = end_date + timedelta(days=15 * 0.002)
    else:
        end_date = time_ref
        start_date = time_ref - timedelta(days=int(span))
        x_min = start_date - timedelta(days=int(span) * 0.002)
        x_max = end_date + timedelta(days=int(span) * 0.002)
        timeRef_deploy=None
        
    logger.info(f"base_ds - start_date: {start_date} end_date {end_date}")
    base_ds = param_data.sel(time=slice(start_date, end_date))
    # drop nans from dataset
    if 'ADCP' not in plot_instrument: # colormesh doesn't accept mask for ADCP
        base_ds = base_ds.where(((base_ds[Yparam].notnull()) & (base_ds[press_param].notnull())).compute(), drop=True)
        plot_func = None # default function is contourf()
        press_label = "Pressure (dbar)"
    else: 
        plot_func = "meshgrid"
        press_label = "Depth (m)"

    static_param = variable_param_dict[param_nickname]['static']
    
    scatter_x = base_ds.time.values
    scatter_y = np.array([])
    scatter_z = np.array([])
    if len(scatter_x) > 5:
        scatter_y = base_ds[press_param].values
        scatter_z = base_ds[Yparam].values
        if 'ADCP' not in plot_instrument:
        # create interpolation grid
            xi, yi, zi, xi_dt = create_interpolation_grid(
                Yparam, 
                press_param, 
                y_min, 
                y_max,
                span, 
                profile_list, 
                logger, 
                unix_epoch, 
                one_second, 
                x_min, 
                x_max, 
                base_ds, 
                scatter_x, 
                scatter_y, 
                scatter_z,
            )
            
            empty_slice, ax = plot_and_save_no_overlay_plots(
                plotter,
                yi,
                zi,
                xi_dt,
                color_map,
                span_string,
                timeRef_deploy,
                fileName_base,
                file_name_list,
                z_min,
                z_max,
                zMin_local,
                zMax_local,
                press_label,
                dpi,
            )
        else: # ADCP routine

                yi = base_ds[press_param].T #transpose
                zi = base_ds[Yparam].T #transpose
                xi_dt = base_ds.time

                empty_slice, ax = plot_and_save_no_overlay_plots(
                    plotter,
                    yi,
                    zi,
                    xi_dt, # can cause problems if nans in x and y coords go into meshgrid
                    color_map,
                    span_string,
                    timeRef_deploy,
                    fileName_base,
                    file_name_list,
                    z_min,
                    z_max,
                    zMin_local,
                    zMax_local,
                    press_label,
                    dpi,
                    plot_func, # use meshgrid in place of contourf for ADCPs due to data density
                    static_param,
                )

    else:
        params = {'range':'full'}
        profilePlot, ax = plotter(0, 0, 0, 'empty', color_map, 'No Data Available', params, press_label)
        file_name = fileName_base + '_' + span_string + '_' + 'none'
        save_fig(profilePlot, file_name_list, file_name, dpi, ['_full', '_standard', '_local'])
        empty_slice = True

    if not empty_slice:
        for overlay in overlays:
            if 'anno' in overlay:
                if overlayData_anno:
                    plot_annotations = {}
                    for i in overlayData_anno:
                        anno_start = datetime.fromtimestamp(int(i['beginDT'])/1000)
                        if i['endDT'] is not None:
                            anno_end = datetime.fromtimestamp(int(i['endDT'])/1000)
                        else:
                            anno_end = i['endDT']
                        in_range,start_anno_line,end_anno_line = annoInRange(start_date,end_date,anno_start,anno_end)
                        if in_range:
                            plot_annotations[start_anno_line] = {'end_anno_line': end_anno_line, 'annotation': i['annotation']}
                    params = {'range':'full'}
                    profilePlot, ax = plotter(xi_dt, yi, zi, 'contour', color_map, 'no', params, press_label, plot_func)
                    if 'deploy' in span_string:
                            plt.axvline(timeRef_deploy,linewidth=1,color='k',linestyle='-.')
                    if plot_annotations:
                        anno_lines = []
                        i = 0
                        for k in plot_annotations.keys():
                            anno_xmin,anno_xmax = annoXnormalize(start_date,end_date,k,plot_annotations[k]['end_anno_line'])
                            A = ax.axhline(y_max,anno_xmin,anno_xmax,linewidth=3,color='r',linestyle='-')
                            A.set_gid(f'anno_{i}')
                            anno_lines.append(A)
                            annotation_string = tw.fill(tw.dedent(plot_annotations[k]['annotation'].rstrip()), width=50)
                            anno_text = ax.annotate(annotation_string,
                                       xy=(k,y_max),xytext=(0.25,0.25), textcoords='axes fraction',
                                       bbox=dict(boxstyle='round',fc='w'),wrap=True,fontsize=5,
                                       zorder = 1, clip_on=True
                            )
                            anno_text.set_gid(f'label_{i}')
                            i += 1
                        f=io.BytesIO()
                        plt.savefig(f, format="svg",dpi=300)
                        file_name = fileName_base + '_' + span_string + '_' + 'anno_full'
                        saveAnnos_SVG(anno_lines,f,file_name)
                    else:
                        file_name = fileName_base + '_' + span_string + '_' + 'anno_full'
                        profilePlot.savefig(file_name + '.png', dpi=300)

                    params = {'range':'standard'}
                    params['vmin'] = z_min
                    params['vmax'] = z_max
                    profilePlot, ax = plotter(xi_dt, yi, zi, 'contour', color_map, 'no', params, press_label, plot_func)
                    if 'deploy' in span_string:
                            plt.axvline(timeRef_deploy,linewidth=1,color='k',linestyle='-.')
                    if plot_annotations:
                        anno_lines = []
                        i = 0
                        for k in plot_annotations.keys():
                            anno_xmin,anno_xmax = annoXnormalize(start_date,end_date,k,plot_annotations[k]['end_anno_line'])
                            A = ax.axhline(y_max,anno_xmin,anno_xmax,linewidth=3,color='r',linestyle='-')
                            A.set_gid(f'anno_{i}')
                            anno_lines.append(A)
                            annotation_string = tw.fill(tw.dedent(plot_annotations[k]['annotation'].rstrip()), width=50)
                            anno_text = ax.annotate(annotation_string,
                                       xy=(k,y_max),xytext=(0.25,0.25), textcoords='axes fraction',
                                       bbox=dict(boxstyle='round',fc='w'),wrap=True,fontsize=5,
                                       zorder = 1, clip_on=True
                            )
                            anno_text.set_gid(f'label_{i}')
                            i += 1
                        f=io.BytesIO()
                        plt.savefig(f, format="svg",dpi=300)
                        file_name = fileName_base + '_' + span_string + '_' + 'anno_standard'
                        saveAnnos_SVG(anno_lines,f,file_name)
                    else:
                        file_name = fileName_base + '_' + span_string + '_' + 'anno_standard'
                        profilePlot.savefig(file_name + '.png', dpi=300)

                    params = {'range':'local'}
                    params['vmin'] = zMin_local
                    params['vmax'] = zMax_local
                    profilePlot, ax = plotter(xi_dt, yi, zi, 'contour', color_map, 'no', params, press_label, plot_func)
                    if 'deploy' in span_string:
                            plt.axvline(timeRef_deploy,linewidth=1,color='k',linestyle='-.')
                    if plot_annotations:
                        anno_lines = []
                        i = 0
                        for k in plot_annotations.keys():
                            anno_xmin,anno_xmax = annoXnormalize(start_date,end_date,k,plot_annotations[k]['end_anno_line'])
                            A = ax.axhline(y_max,anno_xmin,anno_xmax,linewidth=3,color='r',linestyle='-')
                            A.set_gid(f'anno_{i}')
                            anno_lines.append(A)
                            annotation_string = tw.fill(tw.dedent(plot_annotations[k]['annotation'].rstrip()), width=50)
                            anno_text = ax.annotate(annotation_string,
                                       xy=(k,y_max),xytext=(0.25,0.25), textcoords='axes fraction',
                                       bbox=dict(boxstyle='round',fc='w'),wrap=True,fontsize=5,
                                       zorder = 1, clip_on=True
                            )
                            anno_text.set_gid(f'label_{i}')
                            i += 1
                        f=io.BytesIO()
                        plt.savefig(f, format="svg",dpi=300)
                        file_name = fileName_base + '_' + span_string + '_' + 'anno_local'
                        saveAnnos_SVG(anno_lines,f,file_name)
                    else:
                        file_name = fileName_base + '_' + span_string + '_' + 'anno_local'
                        profilePlot.savefig(file_name + '.png', dpi=300)

            if 'clim' in overlay:
                if overlayData_clim:
                    logger.info("clim overlay...")
                    depth_list = []
                    time_list = []
                    clim_list = []
                    for key in overlayData_clim:
                        for sub_key in overlayData_clim[key]:
                            climatology = ast.literal_eval(
                                overlayData_clim[key][sub_key]
                            )
                            clim_list.append(
                                st.mean([climatology[0], climatology[1]])
                            )
                            depth_list.append(ast.literal_eval(sub_key)[0])
                            time_list.append(
                                np.datetime64(
                                    "{0}-{1}-{2}".format(
                                        str(time_ref.year),
                                        str(key).zfill(2),
                                        15,
                                    ),
                                    'D',
                                )
                            )
                            # extend climatology to previous year
                            clim_list.append(
                                st.mean([climatology[0], climatology[1]])
                            )
                            depth_list.append(ast.literal_eval(sub_key)[0])
                            time_list.append(
                                np.datetime64(
                                    "{0}-{1}-{2}".format(
                                        str(time_ref.year - 1),
                                        str(key).zfill(2),
                                        15,
                                    ),
                                    'D',
                                )
                            )
                            # extend climatology to next year
                            clim_list.append(
                                st.mean([climatology[0], climatology[1]])
                            )
                            depth_list.append(ast.literal_eval(sub_key)[0])
                            time_list.append(
                                np.datetime64(
                                    "{0}-{1}-{2}".format(
                                        str(time_ref.year + 1),
                                        str(key).zfill(2),
                                        15,
                                    ),
                                    'D',
                                )
                            )

                    climTime_TS = [
                        ((dt64 - unix_epoch) / one_second) for dt64 in time_list
                    ]
                    # interpolate climatology data
                    logger.info("interpolate climatology data")
                    clim_zi = griddata(
                        (climTime_TS, depth_list),
                        clim_list,
                        (xi, yi),
                        method='linear',
                    )
                    clim_diff = zi - clim_zi
                    if np.isnan(clim_diff).all():
                        logger.info('error gridding climatology, all nans in clim_diff!')
                        params = {'range':'full'}
                        profilePlot, ax = plotter(0, 0, 0, 'empty', color_map, 'Error gridding climatology data', params, press_label)
                        file_name = fileName_base + '_' + span_string + '_' + 'clim'
                        save_fig(profilePlot, file_name_list, file_name, dpi, ['_full', '_standard', '_local'])
              
                    else:
                        max_lim = max(
                            abs(np.nanmin(clim_diff)), abs(np.nanmax(clim_diff))
                        )
                        # plot filled contours
                        clim_params = {}
                        clim_params['range'] = 'na'
                        clim_params['norm'] = 'no'
                        clim_params['vmin'] = -max_lim
                        clim_params['vmax'] = max_lim
                        clim_plot = plotter(xi_dt, yi, clim_diff, 'clim', 'cmo.balance', 'no', clim_params, press_label)
                        if 'deploy' in span_string:
                            plt.axvline(timeRef_deploy,linewidth=1,color='k',linestyle='-.')
                        file_name = fileName_base + '_' + span_string + '_' + 'clim'
                        plt.savefig(file_name + '_full.png', dpi=300)
                        file_name_list.append(file_name + '_full.png')

                        clim_diff_min = np.nanmin(clim_diff)
                        clim_diff_max = np.nanmax(clim_diff)
                        logger.info(f"clim_diff_min: {clim_diff_min} clim_diff_max: {clim_diff_max}")
                        if clim_diff_max < 0:
                            clim_diff_max = 0
                            color_map_standard = balance_blue
                            div_color = 'no'
                        elif clim_diff_min > 0:
                            clim_diff_min = 0
                            color_map_standard = 'cmo.amp'
                            div_color = 'no'
                        else:
                            color_map_standard = 'cmo.balance'
                            div_color = 'yes'
                        if 'yes' in div_color:
                        #    divnorm = colors.TwoSlopeNorm(
                        #        vmin=clim_diff_min, vcenter=0, vmax=clim_diff_max
                        #    )
                            # plot filled contours
                            clim_params = {}
                            clim_params['range'] = 'na'
                            clim_params['norm'] = 'yes'
                            ###clim_params['norm']['divnorm'] = divnorm
                            clim_params['vmin'] = clim_diff_min
                            clim_params['vmax'] = clim_diff_max
                            clim_plot = plotter(xi_dt, yi, clim_diff, 'clim', color_map_standard, 'no', clim_params, press_label)

                        else:
                            # plot filled contours
                            clim_params = {}
                            clim_params['range'] = 'na'
                            clim_params['norm'] = 'no'
                            clim_params['vmin'] = clim_diff_min
                            clim_params['vmax'] = clim_diff_max
                            logger.info("entering clim_plot plotter")
                            clim_plot = plotter(xi_dt, yi, clim_diff, 'clim', color_map_standard, 'no', clim_params, press_label)
                            logger.info("clim_plot successful")

                        if 'deploy' in span_string:
                            plt.axvline(timeRef_deploy,linewidth=1,color='k',linestyle='-.')
                        plt.savefig(file_name + '_standard.png', dpi=300)
                        file_name_list.append(file_name + '_standard.png')
                        plt.savefig(file_name + '_local.png', dpi=300)
                        file_name_list.append(file_name + '_local.png')

                else:
                    logger.info('climatology is empty!')
                    params = {'range':'full'}
                    profilePlot, ax = plotter(0, 0, 0, 'empty', color_map, 'No Climatology Data Available', params, press_label)
                    file_name = fileName_base + '_' + span_string + '_' + 'clim'
                    save_fig(profilePlot, file_name_list, file_name, dpi, ['_full', '_standard', '_local'])


    else:
        params = {'range':'full'}
        logger.info("saving files and created file_name_list...")
        profilePlot,ax = plotter(0, 0, 0, 'empty', color_map, 'No Data Available', params, press_label)
        for overlay in overlays:
            if 'none' not in overlay:
                file_name = fileName_base + '_' + span_string + '_' + overlay
                save_fig(profilePlot, file_name_list, file_name, dpi, ['_full', '_standard', '_local'])

    return file_name_list


def create_interpolation_grid(
    Yparam, 
    press_param, 
    y_min, 
    y_max, 
    span, 
    profile_list, 
    logger, 
    unix_epoch, 
    one_second,  
    x_min, 
    x_max, 
    base_ds, 
    scatter_x, 
    scatter_y, 
    scatter_z,
):
    if y_max > 300:
        profile_depth_grid = 5
    else:
        profile_depth_grid = 0.5 # half meter spacing for depth axis on shallow profilers
    x_min_timestamp = x_min.timestamp()
    x_max_timestamp = x_max.timestamp()
    if profile_list.empty:
        print('profile_list empty...interpolating with old method...')
            # x grid in seconds, with points every 1 hour (3600 seconds)
        xi_arr = np.arange(x_min_timestamp, x_max_timestamp, 3600)
        yi_arr = np.arange(y_min, y_max, profile_depth_grid)
        xi, yi = np.meshgrid(xi_arr, yi_arr)

        scatterX_TS = [((dt64 - unix_epoch) / one_second) for dt64 in scatter_x]

            # interpolate data to grid
        zi = griddata(
                (scatterX_TS, scatter_y), scatter_z, (xi, yi), method='linear'
            )

        xi_dt = xi.astype('datetime64[s]')
            # mask out any time gaps greater than 1 day
        time_gaps = np.where(np.diff(scatterX_TS) > 86400)
        if len(time_gaps[0]) > 1:
            gaps = time_gaps[0]
            for gap in gaps:
                gap_mask = (xi > scatterX_TS[gap]) & (xi < scatterX_TS[gap + 1])
                zi[gap_mask] = np.nan
    else:
        if y_max > 300:
            profile_depth = y_max
        else:
            profile_depth = 190
        xi_arr, yi_arr, zi = gridProfiles(base_ds, press_param, Yparam, profile_list, profile_depth, profile_depth_grid)
        if xi_arr.shape[0] == 1:
            logger.info('error with gridding profiles...interpolating with old method...')
                # x grid in seconds, with points every 1 hour (3600 seconds)
            xi_arr = np.arange(x_min_timestamp, x_max_timestamp, 3600)
                # y grid in meters, with points every 1/2 meter
            yi_arr = np.arange(y_min, y_max, profile_depth_grid)
            xi, yi = np.meshgrid(xi_arr, yi_arr)

            scatterX_TS = [((dt64 - unix_epoch) / one_second) for dt64 in scatter_x]

                # interpolate data to grid
            zi = griddata(
                    (scatterX_TS, scatter_y), scatter_z, (xi, yi), method='linear'
                )
            xi_dt = xi.astype('datetime64[s]')
                # mask out any time gaps greater than 1 day
            time_gaps = np.where(np.diff(scatterX_TS) > 86400)
            if len(time_gaps[0]) > 1:
                gaps = time_gaps[0]
                for gap in gaps:
                    gap_mask = (xi > scatterX_TS[gap]) & (xi < scatterX_TS[gap + 1])
                    zi[gap_mask] = np.nan
        else:
            logger.info('success gridding profiles...')
            xi, yi = np.meshgrid(xi_arr, yi_arr)
                ### filter out profile columns with no data where xi == 0
            zero_mask = np.where(xi_arr == 0)
            zi = np.delete(zi,zero_mask, axis=1)
            xi = np.delete(xi,zero_mask, axis=1)
            yi = np.delete(yi,zero_mask, axis=1)
            xi_dt = xi.astype('datetime64[s]')
            if int(span) > 45:
                gap_threshold = 5
            else:
                gap_threshold = 1
            nan_mask = np.where(np.diff(xi_dt) > timedelta(days=gap_threshold))
            zi[nan_mask] = np.nan
            
        # plot filled contours
    return xi, yi, zi, xi_dt


def plot_and_save_no_overlay_plots(
    plotter,
    yi, 
    zi, 
    xi_dt,
    color_map,
    span_string,
    timeRef_deploy,
    fileName_base,
    file_name_list,
    z_min,
    z_max,
    zMin_local,
    zMax_local,
    press_label,
    dpi,
    plot_func=None,
    static_param=False,
):

    if zi.shape[1] > 1:
        if static_param: # for parameters like percent_beam_good only a single range 0-100 is needed
            params = {'range':'full'}
            profilePlot, ax = plotter(xi_dt, yi, zi, 'contour', color_map, 'no', params, press_label, plot_func)
            if 'deploy' in span_string:
                plt.axvline(timeRef_deploy, linewidth=1, color='k', linestyle='-.')
            file_name = fileName_base + '_' + span_string + '_' + 'none'
            save_fig(profilePlot, file_name_list, file_name, dpi, ['_full', '_standard', '_local'])
            empty_slice = False

        else: # for all other parameters
            params = {'range':'full'}
            profilePlot,ax = plotter(xi_dt, yi, zi, 'contour', color_map, 'no', params, press_label, plot_func)
            if 'deploy' in span_string:
                plt.axvline(timeRef_deploy,linewidth=1,color='k',linestyle='-.')
            file_name = fileName_base + '_' + span_string + '_' + 'none'
            profilePlot.savefig(file_name + '_full.png', dpi=300)
            file_name_list.append(file_name + '_full.png')
            params = {'range':'standard'}
            params['vmin'] = z_min
            params['vmax'] = z_max
            profilePlot,ax = plotter(xi_dt, yi, zi, 'contour', color_map, 'no', params, press_label, plot_func)
            if 'deploy' in span_string:
                plt.axvline(timeRef_deploy,linewidth=1,color='k',linestyle='-.')
            profilePlot.savefig(file_name + '_standard.png', dpi=300)
            file_name_list.append(file_name + '_standard.png')
            params = {'range':'local'}
            params['vmin'] = zMin_local
            params['vmax'] = zMax_local
            profilePlot,ax = plotter(xi_dt, yi, zi, 'contour', color_map, 'no', params, press_label, plot_func)
            if 'deploy' in span_string:
                plt.axvline(timeRef_deploy,linewidth=1,color='k',linestyle='-.')
            profilePlot.savefig(file_name + '_local.png', dpi=300)
            file_name_list.append(file_name + '_local.png')
            empty_slice = False
    else:
        params = {'range':'full'}
        profilePlot, ax = plotter(0, 0, 0, 'empty', color_map, 'Insufficient Profiles Found For Gridding', params, press_label, plot_func,)
        file_name = fileName_base + '_' + span_string + '_' + 'none'
        save_fig(profilePlot, file_name_list, file_name, dpi, ['_full', '_standard', '_local'])
        empty_slice = True
    
    return empty_slice, ax

@task
def plotProfilesScatter(
    Xparam,
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
    fileName_base,
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
    ):
    """Scatter plots for the dashboard's profiler views"""
    plt.ioff()
    dpi=200
    file_name_list = []
    logger=select_logger()
    # Plot Overlays
    overlays = ['anno', 'clim', 'disc', 'flag', 'near', 'none']
    # Data Ranges
    ranges = ['full', 'standard', 'local']
    # Descent Sensors (only on descent)
    descent_samples = ['pco2_seawater','ph_seawater']
    # Drop nans
    logger.info(param_data)
    #param_data = param_data.where(param_data[Xparam].notnull().compute(),drop=True) #TODO 
    # this was commented out for mem bug ask Wendi about bringing in back + consequences?

    y_label = 'pressure, m'
    
    if Xparam in descent_samples:
        profile_start = 'peak'
        profile_end = 'end'
    else:
        profile_start = 'start'
        profile_end = 'peak'

    status_string = status_dict[site]


    def setPlot():

        plt.close('all')
        plt.rcParams["font.family"] = "serif"

        fig, ax = plt.subplots()
        fig.set_size_inches(4, 2)
        fig.patch.set_facecolor('white')
        plt.title(plot_title, fontsize=4, loc='left')
        plt.title(status_string, fontsize=4, fontweight=0, color=status_colors[status_string], loc='right', style='italic' )
        plt.ylabel(y_label, fontsize=4)
        ax.tick_params(direction='out', length=2, width=0.5, labelsize=4)
        ax.ticklabel_format(useOffset=False)
        ax.set_ylim(-y_max, 0)
        
        ax.grid(False)
        return (fig, ax)


    def plotOverlays(overlay, figure_handle,  ax_handle, file_name, time_span):
        file_name = file_name.replace('none',overlay)
        if 'anno' in overlay:
            if overlayData_anno:
                plot_annotations = {}
                for i in overlayData_anno:
                    anno_start = datetime.fromtimestamp(int(i['beginDT'])/1000)
                    if i['endDT'] is not None:
                        anno_end = datetime.fromtimestamp(int(i['endDT'])/1000)
                    else:
                        anno_end = i['endDT']
                    in_range,start_anno_line,end_anno_line = annoInRange(start_date,end_date,anno_start,anno_end)
                    if in_range:
                        plot_annotations[start_anno_line] = {'end_anno_line': end_anno_line, 'annotation': i['annotation']}
                if plot_annotations:
                    anno_lines = []
                    i = 0
                    y_limits = plt.gca().get_ylim()
                    x_limits = plt.gca().get_xlim()
                    j = len(plot_annotations)
                    anno_line_colors = list(colors.cnames.values())[0:j]
                    #anno_line_colors = ['#1f78b4','#a6cee3','#b2df8a','#33a02c','#ff7f00','#fdbf6f','#e31a1c','#fb9a99','#542c2c','#6e409c']
                    for k in plot_annotations.keys():
                        A = ax_handle.axhline(y_limits[0]+3+(i*10),0.75,1,linewidth=3,color=anno_line_colors[i],linestyle='-')
                        A.set_gid(f'anno_{i}')
                        anno_lines.append(A)
                        annotation_string = tw.fill(tw.dedent(plot_annotations[k]['annotation'].rstrip()), width=50)
                        anno_text = ax_handle.annotate(annotation_string,
                                   xy=(x_limits[0],y_limits[0]),xytext=(0.25,0.25), textcoords='axes fraction',
                                   bbox=dict(boxstyle='round',fc='w'),wrap=True,fontsize=5,
                                   zorder = 1, clip_on=True
                        )
                        anno_text.set_gid(f'label_{i}')
                        i += 1
                    f=io.BytesIO()
                    figure_handle.savefig(f, format="svg",dpi=dpi)
                    saveAnnos_SVG(anno_lines,f,file_name)
                    file_name_list.append(file_name + '.svg')
                    for child in ax_handle.get_children():
                        if isinstance(child, matplotlib.text.Annotation):
                            child.remove()
                        if isinstance(child, matplotlib.lines.Line2D):
                            child.remove()
                else:
                    figure_handle.savefig(file_name + '.png', dpi=dpi)
                    file_name_list.append(file_name + '.png')

        elif 'clim' in overlay:
            climatology = {}
            if 'day' in span_string:
                clim_months = []
                clim_months.append(pd.to_datetime(time_span[0]).month)
                climatology = extractClimProfiles(clim_months, overlayData_clim)
            else:
                clim_months = sorted(set(range(pd.to_datetime(time_span[0]).month,(pd.to_datetime(time_span[1]).month) + 1)))
                climatology = extractClimProfiles(clim_months, overlayData_clim)
            if climatology:
                x_limits = plt.gca().get_xlim()
                for clim_month in clim_months:
                    clim_depth = [-x for x in climatology[str(clim_month)]['depth']]
                    clim_line = plt.plot(climatology[str(clim_month)]['clim_data'],clim_depth,
                        '-.',color='r',alpha=0.4,linewidth=0.25)
                plt.xlim(x_limits[0], x_limits[1])
                figure_handle.savefig(file_name + '.png', dpi=dpi)
                file_name_list.append(file_name + '.png')
                for child in ax_handle.get_children():
                    if isinstance(child, matplotlib.lines.Line2D):
                        child.remove()

        elif 'flag' in overlay:
            qc_ds = overlayData_flag.sel(time=slice(time_span[0], time_span[1]))
            qc_ds = retrieve_qc(qc_ds)
             # TODO if homebrew_qartod: block goes here, overwriting qc_ds with homebrew qartod array
            flags = {
                    'qartod_grossRange':{'symbol':'+', 'param':'_qartod_executed_gross_range_test'},
                    'qartod_climatology':{'symbol':'x','param':'_qartod_executed_climatology_test'},
                    #'qartod_summary':{'symbol':'1','param':'_qartod_results'},
                    'qc':{'symbol':'s','param':'_qc_summary_flag'},
                }
            for flag_type in flags.keys():
                flag_string = Xparam + flags[flag_type]['param']
                if flag_string in qc_ds:
                    flag_status = {'fail':{'value':4,'color':'r'}, 'suspect':{'value':3,'color':'y'}}
                    for level in flag_status.keys():
                        flagged_ds = qc_ds.where((qc_ds[flag_string] == flag_status[level]['value']).compute(), drop=True)
                        flag_X = flagged_ds[Xparam].values
                        if len(flag_X) > 0:
                            n = len(flag_X)
                            legend_string = f'{flag_type} {level}: {n} points'
                            flag_Y = flagged_ds[press_param].values
                            flag_line = plt.plot(flag_X,-flag_Y,flags[flag_type]['symbol'],color=flag_status[level]['color'],
                            	    markersize=1,label='%s' % legend_string,
                            	    )
                        else:
                            legend_string = f'{flag_type} {level}: no points flagged'
                            flag_line = plt.plot([0],[0],color='w',markersize=0,label='%s' % legend_string,)
                else:
                    print('no paramters found for ',flag_string)
                    legend_string = f'no {flag_type} flags found'
                    flag_line = plt.plot([0],[0],alpha=0,markersize=0,label='%s' % legend_string,)

            # generating custom legend
            handles, labels = ax.get_legend_handles_labels()
            patches = []
            for handle, label in zip(handles, labels):
                patches.append(
                    mlines.Line2D([],[],color=handle.get_color(),marker=handle.get_marker(),
                            markersize=1,linewidth=0,label=label)
                    )
            legend = ax.legend(handles=patches, loc="upper right", fontsize=3)
            figure_handle.savefig(file_name + '.png', dpi=dpi)
            file_name_list.append(file_name + '.png')
            legend.remove()
            for child in ax_handle.get_children():
                if isinstance(child, matplotlib.lines.Line2D):
                    child.remove()
         
        elif 'disc' in overlay:
            if 'deploy' in span_string:
                if overlayData_disc.empty:
                    print('empty data frame, no discrete samples to plot')
                else:
                    print('adding discrete sample data to plot')
                    for cast in set(overlayData_disc['Cast']):
                        cast_data = overlayData_disc[overlayData_disc['Cast'] == cast]
                        cast_time = cast_data['Start Time [UTC]'].iloc[0]
                        for item in discrete_sample_dict[param]['discreteColumn'].replace("'","").split(","):
                            legend_string = f'{cast}, {cast_time}, {item}'
                            if 'Discrete' in item:
                                disc_line = plt.plot(cast_data[item],-cast_data['CTD Pressure [db]'],color='k',markersize=2,linestyle='None', marker='o',label='%s' % legend_string)
                            elif 'CTD' in item:
                                disc_line = plt.plot(cast_data[item],-cast_data['CTD Pressure [db]'],color='g',markersize=2,linestyle='None', marker='o',label='%s' % legend_string)    
                                #disc_line = plt.plot(cast_data[item],-cast_data['CTD Pressure [db]'],color='g',label='%s' % legend_string)
                        
                        # generating custom legend
                        handles, labels = ax.get_legend_handles_labels()
                        patches = []
                        for handle, label in zip(handles, labels):
                            patches.append(
                            mlines.Line2D([],[],color=handle.get_color(),marker=handle.get_marker(),
                                markersize=1,linewidth=0,label=label)
                            )
                        legend = ax.legend(handles=patches, loc="upper right", fontsize=3)
                        figure_handle.savefig(file_name + '.png', dpi=dpi)
                        file_name_list.append(file_name + '.png')
                        legend.remove()
                        for child in ax_handle.get_children():
                            if isinstance(child, matplotlib.lines.Line2D):
                                child.remove()                         
    
                        
        return



    logger.info(f'plotting profiles for time_span: {span} ')
    profile_iterator = 0    
    if len(profile_list) == 0:
        logger.info('profile_list empty...cannot create profile scatter plots')
        fig,ax = setPlot()
        plt.annotate(
            'No Profile Indices Available', xy=(0.3, 0.5), xycoords='axes fraction'
        )
        file_name = fileName_base + '_' + str(profile_iterator).zfill(3) + 'profile_' + span_string + '_' + 'none'
        save_fig(fig, file_name_list, file_name, dpi, ['_full', '_standard', '_local'])

    else:
        if 'deploy' in span_string:
            deploy_history = loadDeploymentHistory(site)
            deploy_times = listDeployTimes(deploy_history[site])

            timeRef_deploy = deploy_times[0]
            start_date = timeRef_deploy - timedelta(days=15)
            end_date = timeRef_deploy + timedelta(days=15)
            plot_pre = False
            plot_post = False
            pre_text = False
            dataDict_pre = {}
            dataDict_post = {}

            baseDS_pre = param_data.sel(time=slice(start_date, timeRef_deploy))
            if baseDS_pre.time.size !=0:  
                maskStart_pre = baseDS_pre.time[0].values - np.timedelta64(5,'m')
                maskEnd_pre = baseDS_pre.time[-1].values + np.timedelta64(5,'m')
                mask_pre = (profile_list['start'] > maskStart_pre) & (profile_list['end'] <= maskEnd_pre)
                profiles_pre = profile_list.loc[mask_pre]
                if len(profiles_pre) > 0:
                    for index,profile in profiles_pre.iterrows():
                        data_slice = baseDS_pre.sel(time=slice(profile[profile_start], profile[profile_end]))
                        dataDict_pre[profile['peak']] = {}
                        dataDict_pre[profile['peak']]['scatter_x'] = data_slice[Xparam].values
                        dataDict_pre[profile['peak']]['scatter_y'] = -data_slice[press_param].values
                        dataDict_pre[profile['peak']]['scatter_z'] = data_slice.time.values
                    if dataDict_pre:
                        scatterX_pre = np.concatenate( [ subDict['scatter_x'] for subDict in dataDict_pre.values() ] )
                        scatterY_pre = np.concatenate( [ subDict['scatter_y'] for subDict in dataDict_pre.values() ] )
                        scatterZ_pre = np.concatenate( [ subDict['scatter_z'] for subDict in dataDict_pre.values() ] )
                        plot_pre = True
            
            baseDS_post = param_data.sel(time=slice(timeRef_deploy,end_date))
            if baseDS_post.time.size !=0:
                maskStart_post = baseDS_post.time[0].values - np.timedelta64(5,'m')
                maskEnd_post = baseDS_post.time[-1].values + np.timedelta64(5,'m')
                mask_post = (profile_list['start'] > maskStart_post) & (profile_list['end'] <= maskEnd_post)
                profiles_post = profile_list.loc[mask_post]
                if len(profiles_post) > 0:
                    for index, profile in profiles_post.iterrows():
                        data_slice = baseDS_post.sel(time=slice(profile[profile_start], profile[profile_end]))
                        dataDict_post[profile['peak']] = {}
                        dataDict_post[profile['peak']]['scatter_x'] = data_slice[Xparam].values
                        dataDict_post[profile['peak']]['scatter_y'] = -data_slice[press_param].values
                        dataDict_post[profile['peak']]['scatter_z'] = data_slice.time.values
                    if dataDict_post:
                        scatterX_post = np.concatenate( [ subDict['scatter_x'] for subDict in dataDict_post.values() ] )
                        scatterY_post = np.concatenate( [ subDict['scatter_y'] for subDict in dataDict_post.values() ] )
                        scatterZ_post = np.concatenate( [ subDict['scatter_z'] for subDict in dataDict_post.values() ] )
                        plot_post = True
            
            ### Plot all profiles on one plot
            fig, ax = setPlot()
            plot_overlay = False
            if plot_pre:
                plt.scatter(scatterX_pre,scatterY_pre, s=1, c=scatterZ_pre,cmap='Greens', rasterized=True)
                time_string = np.datetime_as_string(scatterZ_pre[0],unit='D') + ' - ' + np.datetime_as_string(scatterZ_pre[-1],unit='D')
                time_string = time_string.replace('T',' ') 
                plt.text(.01, .99, time_string, size=4, color='#12541f', ha='left', va='top', transform=ax.transAxes)
                pre_text = True
            if plot_post:
                plt.scatter(scatterX_post,scatterY_post, s=1, c=scatterZ_post,cmap="Blues", rasterized=True)
                time_string = np.datetime_as_string(scatterZ_post[0],unit='D') + ' - ' + np.datetime_as_string(scatterZ_post[-1],unit='D')
                time_string = time_string.replace('T',' ') 
                if pre_text:
                    plt.text(.01, .90, time_string, size=4, color='#1f78b4', ha='left', va='top', transform=ax.transAxes)
                else:
                    plt.text(.01, .99, time_string, size=4, color='#1f78b4', ha='left', va='top', transform=ax.transAxes)
            if not plot_pre and not plot_post:
                    plt.annotate('No Data Available', xy=(0.3, 0.5), xycoords='axes fraction')
            
            logger.info("saving scatter plots")
            file_name = fileName_base + '_' + str(profile_iterator).zfill(3) + 'profile_' + span_string + '_' + 'none'
            save_fig(fig, file_name_list, file_name, dpi, ['_full'])
            if plot_pre and plot_post:
                time_span = [scatterZ_pre[0],scatterZ_post[-1]]
                plot_overlay = True
            elif plot_pre and not plot_post:
                time_span = [scatterZ_pre[0],scatterZ_pre[-1]]
                plot_overlay = True
            elif not plot_pre and plot_post:
                time_span = [scatterZ_post[0],scatterZ_post[-1]]
                plot_overlay = True
            if plot_overlay:
                overlay_file_name = file_name + '_full'
                for overlay in overlays:
                    plotOverlays(overlay,fig,ax,overlay_file_name,time_span)
            ax.set_xlim(profile_paramMin, profile_paramMax)
            save_fig(fig, file_name_list, file_name, dpi, ['_standard'])
            if plot_overlay:
                overlay_file_name = file_name + '_standard'
                for overlay in overlays:
                    plotOverlays(overlay,fig,ax,overlay_file_name,time_span)
            ax.set_xlim(profile_paramMin_local, profile_paramMax_local)
            save_fig(fig, file_name_list, file_name, dpi, ['_local'])
            if plot_overlay:
                overlay_file_name = file_name + '_local'
                for overlay in overlays:
                    plotOverlays(overlay,fig,ax,overlay_file_name,time_span)


            profile_iterator += 1
            iterList_pre = []
            iterList_post = []
            if dataDict_pre:
                iterList_pre = sorted(set([k.week for k in dataDict_pre.keys()]))
            if dataDict_post:
                iterList_post = sorted(set([k.week for k in dataDict_post.keys()]))
            iter_list = iterList_pre + iterList_post
            pre_text = False
                
            for span_iter in iter_list:
                fig, ax = setPlot()
                plot_overlay = False
                if plot_pre:
                    try:
                        scatterX_pre = np.concatenate( [ dataDict_pre[i]['scatter_x'] for i in dataDict_pre.keys() if (i.week == span_iter) ] )
                        scatterY_pre = np.concatenate( [ dataDict_pre[i]['scatter_y'] for i in dataDict_pre.keys() if (i.week == span_iter) ] )
                        scatterZ_pre = np.concatenate( [ dataDict_pre[i]['scatter_z'] for i in dataDict_pre.keys() if (i.week == span_iter) ] )
                    except ValueError as e:
                        logger.warning(f'ValueError concatenating pre-deploy data for week {span_iter}: {e}')
                        scatterX_pre = []
                    if len(scatterX_pre) > 0:
                        plt.scatter(scatterX_pre,scatterY_pre, s=1, c=scatterZ_pre,cmap='Greens', rasterized=True)
                        time_string = np.datetime_as_string(scatterZ_pre[0],unit='D')
                        plt.text(.01, .99, time_string, size=4, color='#12541f', ha='left', va='top', transform=ax.transAxes)
                        pre_text = True
                if plot_post:
                    try:
                        scatterX_post = np.concatenate( [ dataDict_post[i]['scatter_x'] for i in dataDict_post.keys() if (i.week == span_iter) ] )
                        scatterY_post = np.concatenate( [ dataDict_post[i]['scatter_y'] for i in dataDict_post.keys() if (i.week == span_iter) ] )
                        scatterZ_post = np.concatenate( [ dataDict_post[i]['scatter_z'] for i in dataDict_post.keys() if (i.week == span_iter) ] )
                    except ValueError as e:
                        logger.warning(f'ValueError concatenating post-deploy data for week {span_iter}: {e}')
                        scatterX_post = []
                    if len(scatterX_post) > 0:
                        plt.scatter(scatterX_post,scatterY_post, s=1, c=scatterZ_post,cmap='Blues', rasterized=True)
                        time_string = np.datetime_as_string(scatterZ_post[0],unit='D')
                        if pre_text:
                            plt.text(.01, .90, time_string, size=4, color='#1f78b4', ha='left', va='top', transform=ax.transAxes)
                        else:
                            plt.text(.01, .99, time_string, size=4, color='#1f78b4', ha='left', va='top', transform=ax.transAxes)

                file_name = fileName_base + '_' + str(profile_iterator).zfill(3) + 'profile_' + span_string + '_' + 'none'
                save_fig(fig, file_name_list, file_name, dpi, ['_full'])
                overlay_file_name = file_name + '_full'
                for overlay in overlays:
                    plotOverlays(overlay,fig,ax,overlay_file_name,time_span)
                ax.set_xlim(profile_paramMin, profile_paramMax)
                save_fig(fig, file_name_list, file_name, dpi, ['_standard'])
                overlay_file_name = file_name + '_standard'
                for overlay in overlays:
                    plotOverlays(overlay,fig,ax,overlay_file_name,time_span)
                ax.set_xlim(profile_paramMin_local, profile_paramMax_local)
                save_fig(fig, file_name_list, file_name, dpi, ['_local'])
                overlay_file_name = file_name + '_local'
                for overlay in overlays:
                    plotOverlays(overlay,fig,ax,overlay_file_name,time_span)
                profile_iterator += 1
                
            
        else:
            end_date = time_ref
            start_date = time_ref - timedelta(days=int(span))
            plot_all = False
    
            base_ds = param_data.sel(time=slice(start_date, end_date))
            if base_ds.time.size !=0:
                mask_start = base_ds.time[0].values - np.timedelta64(5,'m')
                mask_end = base_ds.time[-1].values + np.timedelta64(5,'m')
                mask = (profile_list['start'] > mask_start) & (profile_list['end'] <= mask_end)
                profiles = profile_list.loc[mask]
                data_dict = {}
                if len(profiles) > 0:
                    for index, profile in profiles.iterrows():
                        data_slice = base_ds.sel(time=slice(profile[profile_start], profile[profile_end]))
                        data_dict[profile['peak']] = {}
                        data_dict[profile['peak']]['scatter_x'] = data_slice[Xparam].values
                        data_dict[profile['peak']]['scatter_y'] = -data_slice[press_param].values
                        data_dict[profile['peak']]['scatter_z'] = data_slice.time.values
                    if data_dict:
                        scatter_x = np.concatenate( [ subDict['scatter_x'] for subDict in data_dict.values() ] )
                        scatter_y = np.concatenate( [ subDict['scatter_y'] for subDict in data_dict.values() ] )
                        scatter_z = np.concatenate( [ subDict['scatter_z'] for subDict in data_dict.values() ] )
                        plot_all = True
        
                ### Plot all profiles on one plot
                fig, ax = setPlot()
                plot_overlay = False
                if plot_all and len(scatter_z) > 0:
                    if len(profiles) == 1:
                        plt.plot(scatter_x,scatter_y,'.',color='#1f78b4',markersize=1, rasterized=True)
                    else:
                        plt.scatter(scatter_x,scatter_y, s=1, c=scatter_z,cmap='Blues', rasterized=True)
                    if 'day' in span_string:
                        time_string = np.datetime_as_string(scatter_z[0],unit='m') + ' - ' + np.datetime_as_string(scatter_z[-1],unit='m')
                    else:
                        time_string = np.datetime_as_string(scatter_z[0],unit='D') + ' - ' + np.datetime_as_string(scatter_z[-1],unit='D')
                    time_string = time_string.replace('T',' ')    
                    plt.text(.01, .99, time_string, size=4, color='#1f78b4', ha='left', va='top', transform=ax.transAxes)
                    plot_overlay = True
                else:
                    plt.annotate('No Data Available', xy=(0.3, 0.5), xycoords='axes fraction')
                    
                
                file_name = fileName_base + '_' + str(profile_iterator).zfill(3) + 'profile_' + span_string + '_' + 'none'
                save_fig(fig, file_name_list, file_name, dpi, ['_full'])
                if plot_overlay:
                    time_span = [scatter_z[0], scatter_z[-1]]
                    overlay_file_name = file_name + '_full'
                    for overlay in overlays:
                        plotOverlays(overlay,fig,ax,overlay_file_name,time_span)
                ax.set_xlim(profile_paramMin, profile_paramMax)
                save_fig(fig, file_name_list, file_name, dpi, ['_standard'])
                if plot_overlay:
                    overlay_file_name = file_name + '_standard'
                    for overlay in overlays:
                        plotOverlays(overlay,fig,ax,overlay_file_name,time_span)
                ax.set_xlim(profile_paramMin_local, profile_paramMax_local)
                save_fig(fig, file_name_list, file_name, dpi, ['_local'])
                if plot_overlay:
                    overlay_file_name = file_name + '_local'
                    for overlay in overlays:
                        plotOverlays(overlay,fig,ax,overlay_file_name,time_span)

                if data_dict:
                    if 'day' in span_string:
                        profile_iterator += 1
                        for key in sorted(data_dict.keys()):
                            logger.info(f"Processing {key} for day span")
                            if len(data_dict[key]['scatter_z']) > 0:
                                fig, ax = setPlot()
                                plt.plot(data_dict[key]['scatter_x'],data_dict[key]['scatter_y'],'.',color='#1f78b4',markersize=1, rasterized=True)
                                time_string = key.strftime("%Y-%m-%d %H:%M")
                                plt.text(.01, .99, time_string, size=4, color='#1f78b4', ha='left', va='top', transform=ax.transAxes)
                                file_name = fileName_base + '_' + str(profile_iterator).zfill(3) + 'profile_' + span_string + '_' + 'none'
                                save_fig(fig, file_name_list, file_name, dpi, ['_full'])
                                time_span = [data_dict[key]['scatter_z'][0],data_dict[key]['scatter_z'][-1]]
                                overlay_file_name = file_name + '_full'
                                for overlay in overlays:
                                    plotOverlays(overlay,fig,ax,overlay_file_name,time_span)
                                ax.set_xlim(profile_paramMin, profile_paramMax)
                                save_fig(fig, file_name_list, file_name, dpi, ['_standard'])
                                overlay_file_name = file_name + '_standard'
                                for overlay in overlays:
                                    plotOverlays(overlay,fig,ax,overlay_file_name,time_span)
                                ax.set_xlim(profile_paramMin_local, profile_paramMax_local)
                                save_fig(fig, file_name_list, file_name, dpi, ['_local'])
                                overlay_file_name = file_name + '_local'
                                for overlay in overlays:
                                    plotOverlays(overlay,fig,ax,overlay_file_name,time_span)
                                profile_iterator += 1
                    elif 'week' in span_string:
                        profile_iterator += 1 
                        iter_list = [[k.year,k.month,k.day] for k in data_dict.keys()]
                        iterList_sorted = sorted({tuple(i) for i in iter_list}, key=lambda element: (element[0], element[1], element[2]))
                        for span_iter in iterList_sorted:
                            logger.info(f"Concatenating for {span_iter}")
                            logger.info(f"data_dict {data_dict}")
                            fig, ax = setPlot()
                            scatterX_sub = np.concatenate( [ data_dict[i]['scatter_x'] for i in data_dict.keys() if ( (i.day == span_iter[2]) and (i.year == span_iter[0]) and (i.month == span_iter[1]) ) ] )
                            scatterY_sub = np.concatenate( [ data_dict[i]['scatter_y'] for i in data_dict.keys() if ( (i.day == span_iter[2]) and (i.year == span_iter[0]) and (i.month == span_iter[1]) ) ] )
                            scatterZ_sub = np.concatenate( [ data_dict[i]['scatter_z'] for i in data_dict.keys() if ( (i.day == span_iter[2]) and (i.year == span_iter[0]) and (i.month == span_iter[1]) ) ] )
                            if len(scatterZ_sub) > 0:
                                plt.scatter(scatterX_sub,scatterY_sub, s=1, c=scatterZ_sub,cmap='Blues', rasterized=True)
                                time_string = np.datetime_as_string(scatterZ_sub[0],unit='D')
                                plt.text(.01, .99, time_string, size=4, color='#1f78b4', ha='left', va='top', transform=ax.transAxes)
                                file_name = fileName_base + '_' + str(profile_iterator).zfill(3) + 'profile_' + span_string + '_' + 'none'
                                save_fig(fig, file_name_list, file_name, dpi, ['_full'])
                                time_span = [scatterZ_sub[0],scatterZ_sub[-1]]
                                overlay_file_name = file_name + '_full'
                                for overlay in overlays:
                                    plotOverlays(overlay,fig,ax,overlay_file_name,time_span)
                                ax.set_xlim(profile_paramMin, profile_paramMax)
                                save_fig(fig, file_name_list, file_name, dpi, ['_standard'])
                                overlay_file_name = file_name + '_standard'
                                for overlay in overlays:
                                    plotOverlays(overlay,fig,ax,overlay_file_name,time_span)
                                ax.set_xlim(profile_paramMin_local, profile_paramMax_local)
                                save_fig(fig, file_name_list, file_name, dpi, ['_local'])
                                overlay_file_name = file_name + '_local'
                                for overlay in overlays:
                                    plotOverlays(overlay,fig,ax,overlay_file_name,time_span)
                                profile_iterator += 1
                    elif 'month' in span_string:
                        profile_iterator += 1
                        iter_list = [[k.year,k.week] for k in data_dict.keys()]
                        iterList_sorted = sorted({tuple(i) for i in iter_list}, key=lambda element: (element[0], element[1]))
                        for span_iter in iterList_sorted:
                            logger.info(f"Concatenating for {span_iter}")
                            logger.info(f"data_dict {data_dict}")
                            fig, ax = setPlot()
                            scatterX_sub = np.concatenate( [ data_dict[i]['scatter_x'] for i in data_dict.keys() if ( (i.week == span_iter[1]) and (i.year == span_iter[0]) ) ] )
                            scatterY_sub = np.concatenate( [ data_dict[i]['scatter_y'] for i in data_dict.keys() if ( (i.week == span_iter[1]) and (i.year == span_iter[0]) ) ] )
                            scatterZ_sub = np.concatenate( [ data_dict[i]['scatter_z'] for i in data_dict.keys() if ( (i.week == span_iter[1]) and (i.year == span_iter[0]) ) ] )
                            if len(scatterZ_sub) > 0:
                                plt.scatter(scatterX_sub,scatterY_sub, s=1, c=scatterZ_sub,cmap='Blues', rasterized=True)
                                time_string = np.datetime_as_string(scatterZ_sub[0],unit='D') + ' - ' + np.datetime_as_string(scatterZ_sub[-1],unit='D')
                                time_string = time_string.replace('T',' ') 
                                plt.text(.01, .99, time_string, size=4, color='#1f78b4', ha='left', va='top', transform=ax.transAxes)
                                file_name = fileName_base + '_' + str(profile_iterator).zfill(3) + 'profile_' + span_string + '_' + 'none'
                                save_fig(fig, file_name_list, file_name, dpi, ['_full'])
                                time_span = [scatterZ_sub[0],scatterZ_sub[-1]]
                                overlay_file_name = file_name + '_full'
                                for overlay in overlays:
                                    plotOverlays(overlay,fig,ax,overlay_file_name,time_span)
                                ax.set_xlim(profile_paramMin, profile_paramMax)
                                save_fig(fig, file_name_list, file_name, dpi, ['_standard'])
                                overlay_file_name = file_name + '_standard'
                                for overlay in overlays:
                                    plotOverlays(overlay,fig,ax,overlay_file_name,time_span)
                                ax.set_xlim(profile_paramMin_local, profile_paramMax_local)
                                save_fig(fig, file_name_list, file_name, dpi, ['_local'])
                                overlay_file_name = file_name + '_local'
                                for overlay in overlays:
                                    plotOverlays(overlay,fig,ax,overlay_file_name,time_span)
                                profile_iterator += 1
                    elif 'year' in span_string:
                        profile_iterator += 1
                        iter_list = [[k.year,k.month] for k in data_dict.keys()]
                        iterList_sorted = sorted({tuple(i) for i in iter_list}, key=lambda element: (element[0], element[1]))
                        for span_iter in iterList_sorted:
                            logger.info(f"Concatenating for {span_iter}")
                            logger.info(f"data_dict {data_dict}")
                            fig, ax = setPlot()
                            scatterX_sub = np.concatenate( [ data_dict[i]['scatter_x'] for i in data_dict.keys() if ( (i.month == span_iter[1]) and (i.year == span_iter[0]) ) ] )
                            scatterY_sub = np.concatenate( [ data_dict[i]['scatter_y'] for i in data_dict.keys() if ( (i.month == span_iter[1]) and (i.year == span_iter[0]) ) ] )
                            scatterZ_sub = np.concatenate( [ data_dict[i]['scatter_z'] for i in data_dict.keys() if ( (i.month == span_iter[1]) and (i.year == span_iter[0]) ) ] )
                            if len(scatterZ_sub) > 0:
                                plt.scatter(scatterX_sub,scatterY_sub, s=1, c=scatterZ_sub,cmap='Blues', rasterized=True)
                                time_string = np.datetime_as_string(scatterZ_sub[0],unit='D') + ' - ' + np.datetime_as_string(scatterZ_sub[-1],unit='D')
                                time_string = time_string.replace('T',' ') 
                                plt.text(.01, .99, time_string, size=4, color='#1f78b4', ha='left', va='top', transform=ax.transAxes)
                                file_name = fileName_base + '_' + str(profile_iterator).zfill(3) + 'profile_' + span_string + '_' + 'none'
                                save_fig(fig, file_name_list, file_name, dpi, ['_full'])
                                time_span = [scatterZ_sub[0],scatterZ_sub[-1]]
                                overlay_file_name = file_name + '_full'
                                for overlay in overlays:
                                    plotOverlays(overlay,fig,ax,overlay_file_name,time_span)
                                ax.set_xlim(profile_paramMin, profile_paramMax)
                                save_fig(fig, file_name_list, file_name, dpi, ['_standard'])
                                overlay_file_name = file_name + '_standard'
                                for overlay in overlays:
                                    plotOverlays(overlay,fig,ax,overlay_file_name,time_span)
                                ax.set_xlim(profile_paramMin_local, profile_paramMax_local)
                                save_fig(fig, file_name_list, file_name, dpi, ['_local'])
                                overlay_file_name = file_name + '_local'
                                for overlay in overlays:
                                    plotOverlays(overlay,fig,ax,overlay_file_name,time_span)
                                profile_iterator += 1
            else:            
                fig,ax = setPlot()
                plt.annotate(
                    'No Data Available', xy=(0.3, 0.5), xycoords='axes fraction'
                )
                file_name = fileName_base + '_' + str(profile_iterator).zfill(3) + 'profile_' + span_string + '_' + 'none'
                save_fig(fig, file_name_list, file_name, dpi, ['_full', '_standard', '_local'])

    return file_name_list



@task
def plotScatter(
    Yparam,
    param,
    param_data,
    plot_title,
    y_label,
    time_ref,
    y_min,
    y_max,
    yMin_local,
    yMax_local,
    fileName_base,
    overlayData_anno,
    overlayData_clim,
    overlayData_disc,
    overlayData_flag,
    overlayData_near,
    plot_marker_size,
    span,
    span_string,
    status_dict,
    site,
):
    """Scatter plots for the dashboard's fixed depth and colormap (default) view"""
    file_name_list = []
    dpi = 300
    # Plot Overlays
    overlays = ['anno','clim','disc', 'flag', 'near', 'time', 'none']

    # Data Ranges
    ranges = ['full', 'standard', 'local']

    line_colors = [ # we need to extend this every year or find a more permanent solution
        '#1f78b4',
        '#a6cee3',
        '#b2df8a',
        '#33a02c',
        '#ff7f00',
        '#fdbf6f',
        '#e31a1c',
        '#fb9a99',
        '#542c2c',
        '#6e409c',
        '#16f5f5',
        '#A19D9C',
    ]
    balance_big = plt.get_cmap('cmo.balance', 512)
    balance_blue = ListedColormap(balance_big(np.linspace(0, 0.5, 256)))

    status_string = status_dict[site]


    def setPlot():

        plt.close('all')
        plt.rcParams["font.family"] = "serif"

        fig, ax = plt.subplots()
        fig.set_size_inches(5, 1.75)
        fig.patch.set_facecolor('white')
        plt.title(plot_title, fontsize=4, loc='left')
        plt.title(status_string, fontsize=4, fontweight=0, color=status_colors[status_string], loc='right', style='italic' )
        plt.ylabel(y_label, fontsize=4)
        ax.tick_params(direction='out', length=2, width=0.5, labelsize=4)
        ax.ticklabel_format(useOffset=False)
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        formatter.formats = [
            '%y',  # ticks are mostly years
            '%b',  # ticks are mostly months
            '%m/%d',  # ticks are mostly days
            '%H h',  # hrs
            '%H:%M',  # min
            '%S.%f',
        ]  # secs
        formatter.zero_formats = [
            '',  # ticks are mostly years, no need for zero_format
            '%b-%Y',  # ticks are mostly months, mark month/year
            '%m/%d',  # ticks are mostly days, mark month/year
            '%m/%d',  # ticks are mostly hours, mark month and day
            '%H',  # ticks are montly mins, mark hour
            '%M',
        ]  # ticks are mostly seconds, mark minute

        formatter.offset_formats = [
            '',
            '',
            '',
            '',
            '',
            '',
        ]

        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(False)
        return (fig, ax)

    print('plotting scatter for time_span: ', span)

    if 'deploy' in span_string:
        deploy_history = loadDeploymentHistory(site)
        deploy_times = listDeployTimes(deploy_history[site])

        timeRef_deploy = deploy_times[0]
        start_date = timeRef_deploy - timedelta(days=15)
        end_date = timeRef_deploy + timedelta(days=15)
        x_min = start_date - timedelta(days=15 * 0.002)
        x_max = end_date + timedelta(days=15 * 0.002)
    else:
        end_date = time_ref
        start_date = time_ref - timedelta(days=int(span))
        x_min = start_date - timedelta(days=int(span) * 0.002)
        x_max = end_date + timedelta(days=int(span) * 0.002)

    base_ds = param_data.sel(time=slice(start_date, end_date))
    
    scatter_x = base_ds.time.values
    scatter_y = np.array([])
    if len(scatter_x) > 0:
        scatter_y = base_ds.values
    if ('small' in plot_marker_size) & (len(scatter_x) < 1000):
        plot_marker_size = 'medium'
    fig, ax = setPlot()
    empty_slice = False
    if 'large' in plot_marker_size:
        plt.plot(scatter_x, scatter_y, '.', color=line_colors[0], markersize=2)
    elif 'medium' in plot_marker_size:
        plt.plot(scatter_x, scatter_y, '.', color=line_colors[0], markersize=0.75)
    elif 'small' in plot_marker_size:
        plt.plot(scatter_x, scatter_y, ',', color=line_colors[0])
    if 'deploy' in span_string:
        plt.axvline(timeRef_deploy,linewidth=1,color='k',linestyle='-.')
    plt.xlim(x_min, x_max)
    ylim_full = plt.gca().get_ylim()
    if scatter_x.size == 0:
        print('slice is empty!')
        plt.annotate(
            'No Data Available', xy=(0.3, 0.5), xycoords='axes fraction'
        )
        empty_slice = True
        plt.xlim(x_min, x_max)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    for axis in ['top','bottom','left','right']:
        cax.spines[axis].set_linewidth(0)
    cax.set_xticks([])
    cax.set_yticks([])
    file_name = fileName_base + '_' + span_string + '_' + 'none'
    save_fig(fig, file_name_list, file_name, dpi, ['_full'])
    ax.set_ylim(y_min, y_max)
    save_fig(fig, file_name_list, file_name, dpi, ['_standard'])
    ax.set_ylim(yMin_local, yMax_local)
    save_fig(fig, file_name_list, file_name, dpi, ['_local'])

    plot_y_ranges={}
    plot_y_ranges['full']={'y_min': ylim_full[0], 'y_max':ylim_full[1]}
    plot_y_ranges['standard'] = {'y_min': y_min, 'y_max': y_max}
    plot_y_ranges['local'] = {'y_min': yMin_local, 'y_max': yMax_local}


    for overlay in overlays:
        if 'anno' in overlay:
            print('adding annotations to plot')
            for plot_range in ranges:
                plot_ymin = plot_y_ranges[plot_range]['y_min']
                plot_ymax = plot_y_ranges[plot_range]['y_max']
                fig, ax = setPlot()
                if not empty_slice:
                    if 'large' in plot_marker_size:
                        plt.plot(scatter_x, scatter_y, '.', color=line_colors[0], markersize=2, rasterized=True)
                    elif 'medium' in plot_marker_size:
                        plt.plot(scatter_x, scatter_y, '.', color=line_colors[0], markersize=0.75, rasterized=True)
                    elif 'small' in plot_marker_size:
                        plt.plot(scatter_x, scatter_y, ',', color=line_colors[0], rasterized=True)
                    if 'deploy' in span_string:
                        plt.axvline(timeRef_deploy,linewidth=1,color='k',linestyle='-.', rasterized=True)
                    plt.xlim(x_min, x_max)
                if empty_slice:
                    plt.annotate(
                        'No Data Available', xy=(0.3, 0.5), xycoords='axes fraction'
                    )
                    plt.xlim(x_min, x_max)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="2%", pad=0.05)
                for axis in ['top','bottom','left','right']:
                    cax.spines[axis].set_linewidth(0)
                cax.set_xticks([])
                cax.set_yticks([])
                ax.set_ylim(plot_ymin, plot_ymax)
                if overlayData_anno:
                    plot_annotations = {}
                    for i in overlayData_anno:
                        anno_start = datetime.fromtimestamp(int(i['beginDT'])/1000)
                        if i['endDT'] is not None:
                            anno_end = datetime.fromtimestamp(int(i['endDT'])/1000)
                        else:
                            anno_end = i['endDT']
                        in_range,start_anno_line,end_anno_line = annoInRange(start_date,end_date,anno_start,anno_end)
                        if in_range:
                            plot_annotations[start_anno_line] = {'end_anno_line': end_anno_line, 'annotation': i['annotation']}
                    if plot_annotations:
                        anno_lines = []
                        i = 0
                        for k in plot_annotations.keys():
                            anno_xmin,anno_xmax = annoXnormalize(start_date,end_date,k,plot_annotations[k]['end_anno_line'])
                            A = ax.axhline(plot_ymin,anno_xmin,anno_xmax,linewidth=3,color='r',linestyle='-')
                            A.set_gid(f'anno_{i}')
                            anno_lines.append(A)
                            annotation_string = tw.fill(tw.dedent(plot_annotations[k]['annotation'].rstrip()), width=50)
                            anno_text = ax.annotate(annotation_string,
                                       xy=(k,plot_ymin),xytext=(0.25,0.25), textcoords='axes fraction',
                                       bbox=dict(boxstyle='round',facecolor='white'),wrap=True,fontsize=5,
                                       zorder = 20, clip_on=True
                            )
                            anno_text.set_gid(f'label_{i}')
                            i += 1
                        f=io.BytesIO()
                        plt.savefig(f, format="svg",dpi=dpi)
                        file_name = fileName_base + '_' + span_string + '_' + 'anno_' + plot_range
                        saveAnnos_SVG(anno_lines,f,file_name)
                    else:
                        file_name = fileName_base + '_' + span_string + '_' + 'anno_' + plot_range
                        plt.savefig(file_name + '.png', dpi=dpi)
                else:
                    file_name = fileName_base + '_' + span_string + '_' + 'anno_' + plot_range
                    plt.savefig(file_name + '.png', dpi=dpi)


        if 'time' in overlay:
            fig, ax = setPlot()
            
            print('adding time machine plot')
            time_machine_list = []
            if 'deploy' in span_string:
                x_min_time = parser.parse(str(deploy_times[0].year) + '-06-15')
                x_max_time = parser.parse(str(deploy_times[0].year) + '-09-15')
                plt.xlim(x_min_time, x_max_time)
                year_ref = deploy_times[0].year
                for time in deploy_times:
                    start = time - timedelta(days=15)
                    end = time + timedelta(days=15)
                    time_machine_list.append([time,start,end]) 
            else:
                plt.xlim(x_min, x_max)
                year_ref = time_ref.year
                start = time_ref - timedelta(days=int(span))
                time_machine_list.append([time_ref,start,time_ref])
                start_year = pd.to_datetime(param_data['time'].values.min()).year
                num_years = time_ref.year - start_year
                years = np.arange(1,num_years+1,1)
                for year in years:
                    time = time_ref - timedelta(days=int(year*365))
                    start = time - timedelta(days=int(span))
                    end = time
                    time_machine_list.append([time,start,end])
            
            for time_trace in time_machine_list:
                year_diff = int(year_ref) - int(time_trace[0].year)
                time_ds = param_data.sel(time=slice(time_trace[1],time_trace[2]))
                if time_ds.time.size !=0:
                    min_year = pd.to_datetime(time_ds['time'].values.min()).year
                    max_year = pd.to_datetime(time_ds['time'].values.max()).year
                    if min_year != max_year:
                        legend_string = f'{min_year} - {max_year}'
                    else:
                        legend_string = f'{max_year}'
                    time_ds['plotTime'] = time_ds.time + np.timedelta64(timedelta(days=365 * year_diff))
                    time_x = time_ds.plotTime.values
                    time_y = np.array([])
                    if len(time_x) > 0:
                        time_y = time_ds.values
                    c = line_colors[year_diff]
                    if 'large' in plot_marker_size:
                        plt.plot(time_x, time_y,'.',markersize=2,c=c,label='%s' % legend_string,)
                    elif 'medium' in plot_marker_size:
                        plt.plot(time_x,time_y,'.',markersize=0.75,c=c,label='%s' % legend_string,)
                    elif 'small' in plot_marker_size:
                        plt.plot(time_x,time_y,',',c=c,label='%s' % legend_string,)
                    if 'deploy' in span_string:
                        deployTime_plot = time_trace[0] + np.timedelta64(timedelta(days=365 * year_diff))
                        plt.axvline(deployTime_plot,linewidth=1,color=c,linestyle='-.')
                del time_ds
                gc.collect()

            # generating custom legend
            handles, labels = ax.get_legend_handles_labels()
            patches = []
            for handle, label in zip(handles, labels):
                patches.append(
                    mlines.Line2D(
                        [],
                        [],
                        color=handle.get_color(),
                        marker='o',
                        markersize=1,
                        linewidth=0,
                        label=label,
                    )
                )

            legend = ax.legend(handles=patches, loc="upper right", fontsize=3)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            for axis in ['top','bottom','left','right']:
                cax.spines[axis].set_linewidth(0)
            cax.set_xticks([])
            cax.set_yticks([])
            file_name = fileName_base + '_' + span_string + '_' + overlay
            save_fig(fig, file_name_list, file_name, dpi, ['_full'])
            ax.set_ylim(y_min, y_max)
            save_fig(fig, file_name_list, file_name, dpi, ['_standard'])
            ax.set_ylim(yMin_local, yMax_local)
            save_fig(fig, file_name_list, file_name, dpi, ['_local'])

        if 'clim' in overlay:
            # add climatology trace
            print('adding climatology trace to plot')
            if not empty_slice:
                fig, ax = setPlot()
                plt.xlim(x_min, x_max)
                if not overlayData_clim.empty:
                    if 'large' in plot_marker_size:
                        plt.plot(
                            scatter_x,
                            scatter_y,
                            '.',
                            color=line_colors[0],
                            markersize=2,
                        )
                    elif 'medium' in plot_marker_size:
                        plt.plot(
                            scatter_x,
                            scatter_y,
                            '.',
                            color=line_colors[0],
                            markersize=0.75,
                        )
                    elif 'small' in plot_marker_size:
                        plt.plot(scatter_x, scatter_y, ',', color=line_colors[0])

                    plt.fill_between(
                        overlayData_clim.index,
                        overlayData_clim.clim_minus_3std,
                        overlayData_clim.clim_plus_3std,
                        alpha=0.2,
                    )
                    plt.plot(
                        overlayData_clim.clim_data,
                        '-.',
                        color='r',
                        alpha=0.4,
                        linewidth=0.25,
                    )
                else:
                    print('Climatology is empty!')
                    plt.annotate(
                        'No Climatology Data Available',
                        xy=(0.3, 0.5),
                        xycoords='axes fraction',
                    )
                if 'deploy' in span_string:
                    plt.axvline(timeRef_deploy,linewidth=1,color='k',linestyle='-.')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="2%", pad=0.05)
                for axis in ['top','bottom','left','right']:
                    cax.spines[axis].set_linewidth(0)
                cax.set_xticks([])
                cax.set_yticks([])
                file_name = fileName_base + '_' + span_string + '_' + 'clim'
                save_fig(fig, file_name_list, file_name, dpi, ['_full'])
                ax.set_ylim(y_min, y_max)
                save_fig(fig, file_name_list, file_name, dpi, ['_standard'])
                ax.set_ylim(yMin_local, yMax_local)
                save_fig(fig, file_name_list, file_name, dpi, ['_local'])

        if 'near' in overlay:
            # add nearest neighbor data traces
            print('adding nearest neighbor data to plot')

        if 'disc' in overlay:
            if 'deploy' in span_string:
                if overlayData_disc.empty:
                    print('empty data frame, no discrete samples to plot')
                else:
                    print('adding discrete sample data to plot')
                    for cast in set(overlayData_disc['Cast']):
                        cast_data = overlayData_disc[overlayData_disc['Cast'] == cast]
                        cast_time = cast_data['Start Time [UTC]'].iloc[0]
                        for item in discrete_sample_dict[param]['discreteColumn'].replace("'","").split(","):
                            legend_string = f'{cast}, {cast_time}, {item}'
                            if 'Discrete' in item:
                                disc_line = plt.plot(cast_data[item],-cast_data['CTD Pressure [db]'],color='k',markersize=2,linestyle='None', marker='o',label='%s' % legend_string)
                            elif 'CTD' in item:
                                disc_line = plt.plot(cast_data[item],-cast_data['CTD Pressure [db]'],color='g',markersize=2,linestyle='None', marker='o',label='%s' % legend_string)
                                #disc_line = plt.plot(cast_data[item],-cast_data['CTD Pressure [db]'],color='g',label='%s' % legend_string)
            
                        # generating custom legend
                        handles, labels = ax.get_legend_handles_labels()
                        patches = []
                        for handle, label in zip(handles, labels):
                            patches.append(
                            mlines.Line2D([],[],color=handle.get_color(),marker=handle.get_marker(),
                                markersize=1,linewidth=0,label=label)
                            )
                        legend = ax.legend(handles=patches, loc="upper right", fontsize=3)
                        
                        plt.axvline(timeRef_deploy,linewidth=1,color='k',linestyle='-.')
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="2%", pad=0.05)
                        for axis in ['top','bottom','left','right']:
                            cax.spines[axis].set_linewidth(0)
                            cax.set_xticks([])
                            cax.set_yticks([])
                        file_name = fileName_base + '_' + span_string + '_' + 'disc' 
                        save_fig(fig, file_name_list, file_name, dpi, ['_full'])
                        ax.set_ylim(y_min, y_max)
                        save_fig(fig, file_name_list, file_name, dpi, ['_standard'])
                        ax.set_ylim(yMin_local, yMax_local)
                        save_fig(fig, file_name_list, file_name, dpi, ['_local'])
                        


        if 'flag' in overlay:
            # highlight flagged data points
            print('adding flagged data overlay to plot')
            if not empty_slice:
                fig, ax = setPlot()
                plt.xlim(x_min, x_max)
                legend_string = 'all data'
                if 'large' in plot_marker_size:
                    flag_marker = 3
                    plt.plot(
                        scatter_x, 
                        scatter_y,
                        '.',
                        color=line_colors[0],
                        markersize=2,
                        label='%s' % legend_string,
                    )
                elif 'medium' in plot_marker_size:
                    flag_marker = 1.5
                    plt.plot(
                        scatter_x,
                        scatter_y,
                        '.',
                        color=line_colors[0],
                        markersize=0.75,
                        label='%s' % legend_string,
                    )
                elif 'small' in plot_marker_size:
                    flag_marker = 0.25
                    plt.plot(scatter_x, scatter_y, ',', color=line_colors[0], label='%s' % legend_string,)
                # slice overlayData_flag so we're not working on whole dataset
                qc_ds = overlayData_flag.sel(time=slice(start_date, end_date))
                print(qc_ds)
                # retrieve flags
                qc_ds = retrieve_qc(qc_ds)
                # TODO if homebrew_qartod: block goes here, overwriting qc_ds with homebrew qartod array
                flags = {
                    'qartod_grossRange':{'symbol':'+', 'param':'_qartod_executed_gross_range_test'},
                    'qartod_climatology':{'symbol':'x','param':'_qartod_executed_climatology_test'},
                    #'qartod_summary':{'symbol':'1','param':'_qartod_results'},
                    'qc':{'symbol':'s','param':'_qc_summary_flag'},
                }
                for flag_type in flags.keys():
                    flag_string = Yparam + flags[flag_type]['param']
                    print(flag_string)
                    if flag_string in qc_ds:
                        print(f'paramters found for {flag_string}')
                        flag_status = {'fail':{'value':4,'color':'r'}, 'suspect':{'value':3,'color':'y'}}
                        for level in flag_status.keys():
                            flagged_ds = qc_ds.where((qc_ds[flag_string] == flag_status[level]['value']).compute(), drop=True) # find where the flags DS matches a certain flag status
                            flag_X = flagged_ds.time.values # flagged times
                            if len(flag_X) > 0:
                                n = len(flag_X)
                                legend_string = f'{flag_type} {level}: {n} points'
                                flag_Y = flagged_ds[Yparam].values
                                plt.plot(
                                    flag_X, # flagged times 
                                    flag_Y, # flagged parameter values
                            	    flags[flag_type]['symbol'],
                            	    color=flag_status[level]['color'],
                            	    markersize=flag_marker,
                            	    label='%s' % legend_string,   
                            	    )
                            else:
                                legend_string = f'{flag_type} {level}: no points flagged'
                                plt.plot([0],[0],color='w',markersize=0,label='%s' % legend_string,)
                    else:
                        print('no paramters found for ',flag_string)
                        legend_string = f'no {flag_type} flags found'
                        plt.plot(scatter_x,scatter_y,alpha=0,markersize=0,label='%s' % legend_string,)

                # generating custom legend 
                handles, labels = ax.get_legend_handles_labels()
                patches = []
                for handle, label in zip(handles, labels):
                    patches.append(
                        mlines.Line2D(
                            [],  
                            [],
                            color=handle.get_color(),
                            marker=handle.get_marker(),
                            markersize=1,
                            linewidth=0,  
                            label=label,
                        )
                    )
                  
                legend = ax.legend(handles=patches, loc="upper right", fontsize=3)


                if 'deploy' in span_string:
                    plt.axvline(timeRef_deploy,linewidth=1,color='k',linestyle='-.')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="2%", pad=0.05)
                for axis in ['top','bottom','left','right']:
                    cax.spines[axis].set_linewidth(0)
                cax.set_xticks([])
                cax.set_yticks([])
                file_name = fileName_base + '_' + span_string + '_' + 'flag'
                save_fig(fig, file_name_list, file_name, dpi, ['_full'])
                ax.set_ylim(y_min, y_max)
                save_fig(fig, file_name_list, file_name, dpi, ['_standard'])
                ax.set_ylim(yMin_local, yMax_local)
                save_fig(fig, file_name_list, file_name, dpi, ['_local'] )

    return file_name_list



def retrieve_qc(ds):
    """
    Extract the QC test results from the different variables in the data set,
    and create a new variable with the QC test results set to match the logic
    used in QARTOD testing. Instead of setting the results to an integer
    representation of a bitmask, use the pass = 1, not_evaluated = 2,
    suspect_or_of_high_interest = 3, fail = 4 and missing = 9 flag values from
    QARTOD.
    The QC portion of this code was copied from the ooi-data-explorations parse_qc function, 
    which was was inspired by an example notebook developed by the OOI Data
    Team for the 2018 Data Workshops. The original example, by Friedrich Knuth,
    and additional information on the original OOI QC algorithms can be found
    at:
    https://oceanobservatories.org/knowledgebase/interpreting-qc-variables-and-results/
    :param ds: dataset with *_qc_executed and *_qc_results variables
               as well as qartod_executed variables if available
    :return ds: dataset with the *_qc_executed and *_qc_results variables
        reworked to create a new *_qc_summary variable with the results
        of the QC checks decoded into a QARTOD style flag value, as well as 
        extracted qartod variables (gross range and climatology).  Code will need to be
        adapted as more tests are added...
    """
    # create a list of the variables that have had QC tests applied
    variables = [x.split('_qc_results')[0] for x in ds.variables if 'qc_results' in x]

    # for each variable with qc tests applied
    for var in variables:
        # set the qc_results and qc_executed variable names and the new qc_flags variable name
        qc_result = var + '_qc_results'
        qc_executed = var + '_qc_executed'
        qc_summary = var + '_qc_summary_flag'

        # create the initial qc_flags array
        flags = np.tile(np.array([0, 0, 0, 0, 0, 0, 0, 0]), (len(ds.time), 1))
        # the list of tests run, and their bit positions are:
        #    0: dataqc_globalrangetest
        #    1: dataqc_localrangetest
        #    2: dataqc_spiketest
        #    3: dataqc_polytrendtest
        #    4: dataqc_stuckvaluetest
        #    5: dataqc_gradienttest
        #    6: undefined
        #    7: dataqc_propagateflags

        # use the qc_executed variable to determine which tests were run, and set up a bit mask to pull out the results
        executed = np.bitwise_or.reduce(ds[qc_executed].values.astype('uint8'))
        executed_bits = np.unpackbits(executed.astype('uint8'))

        # for each test executed, reset the qc_flags for pass == 1, suspect == 3, or fail == 4
        for index, value in enumerate(executed_bits[::-1]):
            if value:
                if index in [2, 3, 4, 5, 6, 7]:
                    # mark these tests as missing since they are problematic
                    flag = 9
                else:
                    # only mark the global range test as fail, all the other tests are problematic
                    flag = 4
                mask = 2 ** index
                m = (ds[qc_result].values.astype('uint8') & mask) > 0
                flags[m, index] = 1   # True == pass
                flags[~m, index] = flag  # False == suspect/fail

        # add the qc_flags to the dataset, rolling up the results into a single value
        ds[qc_summary] = ('time', flags.max(axis=1, initial=1).astype(np.int32))

    ## create a list of the variables that have had QARTOD tests applied
    ##variables = [x.split('_qartod_executed')[0] for x in ds.variables if 'qartod_executed' in x]

    ### for each variable with qc tests applied
    ##for var in variables:
    ##    qartodString = var + '_qartod_executed'
    ##    flagNameBase = var + '_qartod_'
    ##    testOrder = ds[qartodString][0].tests_executed.strip("'").replace(" ","").split(',')
    ##    for i in range(0, len(testOrder)):
    ##        flag_string = testOrder[i]
    ##        flagIndex = testOrder.index(flag_string)
    ##        flagName = flagNameBase + flag_string
    ##        ds[flagName] = [int(i[flagIndex]) for i in ds[qartodString].values.tolist()]

    return ds


