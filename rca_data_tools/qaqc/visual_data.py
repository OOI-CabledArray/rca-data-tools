import requests
import re
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from dateutil import parser
from loguru import logger
from prefect import task

from rca_data_tools.qaqc.constants import SPAN_DICT, CAM_URL_DICT, N_EXPECTED_IMGS
from rca_data_tools.qaqc.plots import stage3_dict, plotDir, PLOT_DIR 
from rca_data_tools.qaqc.utils import select_logger
# see plots module for directories definitions #TODO neater way to do this?



def extract_numeric(value, full_url):
    match = re.search(r'\d+', value)
    if match:
        if int(match.group()) > 40:
            logger.warning(f"An unusual image size: {value} @ {full_url}")
        return int(match.group())
    else:
        return np.nan
    

def create_daily_cam_df(base_url, str_date, img_size_cutoff):
    logger = select_logger()

    full_url = f"{base_url}{str_date}"
    img_data_list = []
    # parse the string into a datetime object
    parsed_date = datetime.strptime(str_date, '%Y/%m/%d/')
    
    response = requests.get(full_url)
    if response.status_code == 404:
        logger.info(f"404 response for date: {parsed_date}")
        return None

    else:
    
        soup = BeautifulSoup(response.text, "html.parser")
        camds_a_tags = soup.find_all('a', href=lambda href: href and 'CAMDS' in href)
        
        # extract and print the text content of each matching <a> tag
        for a_tag in camds_a_tags[1:]:
            img_name = a_tag.get_text(strip=True)
            text_content = a_tag.find_next_sibling(string=True)
            date_and_size = text_content.strip().split(None, 2)
        
            if len(date_and_size) == 3:
                size = date_and_size[2]
            else:
                size = np.nan
        
            img_data = {"img_name": img_name, "size":size, "date_taken": parsed_date}
            img_data_list.append(img_data)
            
        day_df = pd.DataFrame(img_data_list)
        day_df["size_int"] = day_df["size"].apply(extract_numeric, full_url=full_url)
        # create a categorical variables `possibly blank` if img size is less than a cutoff specific to each cam
        day_df["image_status"] = day_df['size_int'].apply(lambda x: 'possibly_blank' if x < img_size_cutoff else 'not_blank')
    
    return(day_df)


def make_timerange_df(start_date, end_date, base_url, img_size_cutoff):
    date = start_date
    df_list = []

    while date <= end_date:
        str_date = date.strftime('%Y/%m/%d/')
    
        single_day_df = create_daily_cam_df(base_url, str_date, img_size_cutoff)
        df_list.append(single_day_df)
        date += timedelta(days=1)

    full_month_df = pd.concat(df_list)

    unique_sizes = np.unique(full_month_df["size_int"])
    logger.info(f"Images in the scanned range have the following sizes: {unique_sizes}")
    return full_month_df


def make_wide_summary_df(timerange_df, img_size_cutoff):
    summary_df = timerange_df[['date_taken', 'image_status']]
    summary_df = summary_df.groupby(["date_taken", "image_status"])["date_taken"].count().reset_index(name="count")

    wide_df = summary_df.pivot(index='date_taken', columns="image_status", values='count').reset_index()

    if 'not_blank' not in wide_df.columns:
        logger.warning(f"no `not_blank` images found at cutoff {img_size_cutoff} - filling wide df column with zeros")
        wide_df['not_blank'] = 0 
    if 'possibly_blank' not in wide_df.columns:
        logger.warning(f"no `possibly_blank` images found at cutoff {img_size_cutoff} - filling wide df column with zeros")
        wide_df['possibly_blank'] = 0 

    print(wide_df.head())

    return wide_df


@task
def cam_qaqc_stacked_bar(site, time_string, span):

    logger = select_logger()
    PLOT_DIR.mkdir(exist_ok=True)

    plot_list = []
    overlay = 'none'
    depth = 'full' #TODO
    span_str = SPAN_DICT[span]
    file_name = f'{plotDir}{site}_{span_str}_{overlay}_{depth}.png'
    logger.warning(file_name)
    img_size_cutoff = stage3_dict[site]['dataParameters']  #TODO *hacky* consult Wendi

    # calculate time window
    end_date = parser.parse(time_string)
    days_delta = timedelta(days=int(span))
    start_date = end_date - days_delta

    base_url = CAM_URL_DICT[site]
    
    timerange_df = make_timerange_df(start_date, end_date, base_url, img_size_cutoff)
    wide_df = make_wide_summary_df(timerange_df, img_size_cutoff)

    plt.figure(figsize=(20,6))
    plt.bar(wide_df['date_taken'], wide_df['not_blank'], color="blue")
    plt.bar(wide_df['date_taken'], wide_df['possibly_blank'], bottom=wide_df['not_blank'], color="red")
    plt.title(site)
    plt.ylabel("Number of Images")
    plt.axhline(y=N_EXPECTED_IMGS, color='black', linestyle='--')
    plt.axvline(x=end_date, color='black', linewidth=0.5)

    # custom labels and handles legend
    handles = [
    Line2D([0], [0], color='red', linewidth=8),
    Line2D([0], [0], color='blue', linewidth=8)
    ]

    labels = [f'Under {str(img_size_cutoff)}M', f'Over {str(img_size_cutoff)}M']
    plt.legend(handles=handles, labels=labels, loc='upper left')

    # saving plot 
    plt.savefig(file_name)
    plot_list.append(file_name)

    return plot_list

    