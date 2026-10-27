import humanfriendly
import requests
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from dateutil import parser
from loguru import logger
from prefect import task

from rca_data_tools.qaqc.constants import (
    SPAN_DICT,
    CAM_URL_DICT,
    N_EXPECTED_IMGS,
    PLOT_DIR,
    PLOT_DIR_STR,
    STAGE3_DICT,
    STATUS_COLORS,
)
from rca_data_tools.qaqc.dashboard import blankCax, conciseDateAxis, loadStatus
from rca_data_tools.qaqc.utils import select_logger


def extract_numeric(value: str, full_url: str):
    try:
        size_in_bytes = humanfriendly.parse_size(value)
        size_in_mb = size_in_bytes / (1024 * 1024)
        if size_in_mb > 20 and 'CAMDS' in full_url:
            logger.warning(f"An unusual image size: {value} @ {full_url}")
        if size_in_mb < 10000 and 'CAMHD' in full_url:
            logger.warning(f"An unusual image size: {value} @ {full_url}")
        return size_in_mb
    except humanfriendly.InvalidSize:
        logger.warning(f"Invalid file size: {value} @ {full_url}")
        return np.nan


def create_daily_cam_df(base_url: str, str_date: str, img_size_cutoff: int): #cutoff in Mb
    logger = select_logger()

    full_url = f"{base_url}{str_date}"
    img_data_list = []
    # parse the string into a datetime object
    parsed_date = datetime.strptime(str_date, "%Y/%m/%d/")

    response = requests.get(full_url)
    if response.status_code == 404:
        logger.info(f"404 response for date: {parsed_date}")
        return None

    else:

        soup = BeautifulSoup(response.text, "html.parser")
        camds_a_tags = soup.find_all("a", href=lambda href: href and "CAM" in href)

        # extract and print the text content of each matching <a> tag
        for a_tag in camds_a_tags[1:]:
            img_name = a_tag.get_text(strip=True)
            text_content = a_tag.find_next_sibling(string=True)
            date_and_size = text_content.strip().split(None, 2)

            if len(date_and_size) == 3:
                size = date_and_size[2]
            else:
                size = np.nan

            img_data = {"img_name": img_name, "size": size, "date_taken": parsed_date}
            if 'CAMHD' in base_url: # we don't want md5 and mp4 files in the df
                if 'mp4' not in img_data["img_name"] and 'md5' not in img_data["img_name"]:
                    img_data_list.append(img_data)
            else:
                img_data_list.append(img_data)

        # day folder exists but has no images yet (e.g. today, before the first upload)
        if not img_data_list:
            logger.info(f"no images listed for date: {parsed_date}")
            return None

        day_df = pd.DataFrame(img_data_list)
        day_df["size_int"] = day_df["size"].apply(extract_numeric, full_url=full_url)
        # create a categorical variables `possibly blank` if img size is less than a cutoff specific to each cam
        day_df["image_status"] = day_df["size_int"].apply(
            lambda x: "possibly_blank" if x < img_size_cutoff else "not_blank"
        )

    return day_df


def make_timerange_df(start_date, end_date, base_url, img_size_cutoff):
    date = start_date
    df_list = []

    while date <= end_date:
        str_date = date.strftime("%Y/%m/%d/")

        single_day_df = create_daily_cam_df(base_url, str_date, img_size_cutoff)
        df_list.append(single_day_df)
        date += timedelta(days=1)

    try:
        full_month_df = pd.concat(df_list)
    except ValueError:
        logger.warning('No images found in the scanned range.')
        return None

    unique_sizes = np.unique(full_month_df["size_int"])
    logger.info(f"Images in the scanned range have the following sizes: {unique_sizes}")
    return full_month_df


def make_wide_summary_df(timerange_df, img_size_cutoff):
    if timerange_df is None:
        logger.warning('No images found in the scanned range.')
        return None
    else:
        summary_df = timerange_df[["date_taken", "image_status"]]
        summary_df = (
            summary_df.groupby(["date_taken", "image_status"])["date_taken"]
            .count()
            .reset_index(name="count")
        )

        wide_df = summary_df.pivot(
            index="date_taken", columns="image_status", values="count"
        ) .fillna(0).reset_index()

        if "not_blank" not in wide_df.columns:
            logger.warning(
                f"no `not_blank` images found at cutoff {img_size_cutoff} - filling wide df column with zeros"
            )
            wide_df["not_blank"] = 0
        if "possibly_blank" not in wide_df.columns:
            logger.warning(
                f"no `possibly_blank` images found at cutoff {img_size_cutoff} - filling wide df column with zeros"
            )
            wide_df["possibly_blank"] = 0

        return wide_df


@task
def cam_qaqc_stacked_bar(site, time_string, span):
    plt.switch_backend("Agg")

    logger = select_logger()
    PLOT_DIR.mkdir(exist_ok=True)

    plot_list = []
    overlay = "none"
    depth = "full"  # HACK but has been working forever
    span_str = SPAN_DICT[span]
    file_name = f"{PLOT_DIR_STR}{site}_{span_str}_{overlay}_{depth}.png"
    logger.info(file_name)
    img_size_cutoff = int(STAGE3_DICT[site]["dataParameters"])  # HACK probably a cleaner way to config

    # calculate time window
    end_date = parser.parse(time_string)
    days_delta = timedelta(days=int(span))
    start_date = end_date - days_delta

    base_url = CAM_URL_DICT[site]

    timerange_df = make_timerange_df(start_date, end_date, base_url, img_size_cutoff)
    wide_df = make_wide_summary_df(timerange_df, img_size_cutoff)

    # same frame as the instrument plots (dashboard.plotScatter): 5 x 1.75 in at
    # 300 dpi, serif, title left and operational status right, concise dates
    statusString = loadStatus().get(site, 'UNAVAILABLE')
    plt.close('all')
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 1.75)
    fig.patch.set_facecolor('white')
    plt.title(f"{site} images", fontsize=4, loc='left')
    plt.title(statusString, fontsize=4, fontweight=0, color=STATUS_COLORS[statusString], loc='right', style='italic')
    plt.ylabel("Number of Files", fontsize=4)
    ax.tick_params(direction='out', length=2, width=0.5, labelsize=4)
    conciseDateAxis(ax)
    ax.grid(False)

    notBlank, possiblyBlank = '#1f78b4', '#e31a1c'
    if wide_df is not None:
        # each bar covers its own day, starting at midnight
        days = pd.to_datetime(wide_df["date_taken"])
        plt.bar(days, wide_df["not_blank"], width=0.8, align='edge', color=notBlank)
        plt.bar(days, wide_df["possibly_blank"], width=0.8, align='edge',
                bottom=wide_df["not_blank"], color=possiblyBlank)
        if 'CAMDS' in site:
            plt.axhline(y=N_EXPECTED_IMGS, color="k", linewidth=0.5, linestyle="--")
        plt.axvline(x=end_date, color="k", linewidth=1, linestyle="-.")
    else:
        logger.warning("Creating blank plot to flag missing data")
        plt.annotate("No Data Available", xy=(0.3, 0.5), xycoords="axes fraction")
    plt.xlim(start_date, end_date + timedelta(days=1))

    swatch = lambda c, label: Line2D([], [], color=c, marker='s', markersize=2, linewidth=0, label=label)
    ax.legend(handles=[swatch(possiblyBlank, f"Under {img_size_cutoff}Mb"), swatch(notBlank, f"Over {img_size_cutoff}Mb")],
              loc="upper right", fontsize=3)
    blankCax(ax)

    fig.savefig(file_name, dpi=300)
    plot_list.append(file_name)

    return plot_list
