"""discrete.py

This module contains code for compiling and processing discrete bottle samples.

"""
import datetime as dt
import io
import numpy as np
import pandas as pd
from os import path
import os
import re
import requests
from cmislib.model import CmisClient
import xarray as xr
from pathlib import Path

HERE = Path(__file__).parent.absolute()
PARAMS_DIR = HERE.joinpath('params')

def parseDiscrete(io_Object, template):
    try:
        df = pd.read_csv(io_Object,dtype=str)
        
    except Exception:
        print('unable to parse object!')
        df = None
    
    if df is not None:
        df = validateDiscrete(df,template)
    
    return df


def validateDiscrete(df,template):
    
    validate = 1
    
    ### Check for empty cells
    empty_cells = np.where(pd.isnull(df))
    if empty_cells[0].size > 0:
        for cell in empty_cells:
            print('empty cell at ' + str(cell[0]) + ', ' + str(cell[1]))
            validate = 0
            
    ### Verify all column header labels and order match template
    if len(template.keys()) != len(df.keys()) and len(template.keys()) != sum([1 for i, j in zip(template.keys(), df.keys()) if i == j]): 
        print ("Headers do not match template") 
        validate = 0
        
    for header in df.keys():
        if header not in template.keys():
            if header == 'Calculated bicarb [umol/kg]':
                print('fix this file header eventually...', header)
                df.rename(columns={"Calculated bicarb [umol/kg]": "Calculated Bicarb [umol/kg]"},inplace=True)
            elif 'Cruise' in header:
                print('fix this file header eventually...', header)
                df.rename(columns={header: 'Cruise'},inplace=True)
            else:
                print("Header in file not in template: ", header)
                validate = 0
            
    for index,row in df.iterrows():
        for header in df.keys():
            ### Verify all fill values are '-9999999'
            if '-99' in row[header]:
                if not re.search(r'^-9999999$', row[header]):
                    print('fill value improperly formatted: ' + header + ': ' + row[header] + ', row ' + str(index+1))
                    validate = 0
                next
            else:
                ### Identify any flags in non-flag columns
                if re.search(r'^\*[0-1]{16}$',row[header]) and 'Flag' not in header:
                    print(header + ' includes misplaced flag: ' + row[header] + ', row ' + str(index+1))
                    validate = 0
                ###  Verify each flag has an asterix and a 16-character combination of zeroes and ones       
                if 'Flag' in header:
                    if re.search(r'^\*[0]{16}$',row[header]):
                        print(header + ' is a blank flag: ' + row[header] + ', row ' + str(index+1))
                        validate = 0
                    if not re.search(r'^\*[0-1]{16}$',row[header]):
                        print(header + ' not formatted correctly: ' + row[header] + ', row ' + str(index+1))
                        validate = 0
                ### Verify each time is formatted as "YYYY-MM-DDTHH:mm:ss.000Z" and is within cruise dates
                if 'Time' in header:
                    if not re.search(r'^\d\d\d\d-\d\d-\d\dT\d\d:\d\d:\d\d.\d\d\dZ',row[header]):
                        print(header + ' not formatted correctly: ' + row[header] + ', row ' + str(index+1))
                        validate = 0
    if validate == 0:
        df = None
        
    return df

def loadDiscreteData():
    
    client = CmisClient('http://alfresco.oceanobservatories.org/alfresco/s/cmis',os.environ.get("ALF_USER"),os.environ.get("ALF_TOKEN"))
    repo = client.defaultRepository
    results = repo.query("SELECT cmis:name, cmis:objectId FROM cmis:document WHERE CONTAINS('cmis:name:discrete_summary')")
    
    template = pd.read_csv(PARAMS_DIR.joinpath('discreteSummary_template.csv'), dtype=str)

    pathKeys_ALL=['Cruise Data','Water Sampling']
    pathKeys_ANY=['Ship Data','Ship_Data','Shipboard Data']

    df_data = []
    for result in results:
        if ('.csv' in result.name) and ('Cabled-' in result.name):
            data = repo.getObject(result)
            data_string = data.getContentStream()
            path = data.getPaths()[0]
            if all(keyword in path for keyword in pathKeys_ALL) and (any(keyword in path for keyword in pathKeys_ANY)):
                print(result.name)
                df = pd.read_csv(data_string,na_values=['-9999999']) 
                for header in df.keys():
                    if 'Cruise' in header:
                        print('fix this file header eventually...', header)
                        df.rename(columns={header: 'Cruise'},inplace=True)
                df_data.append(df)
                                                                             
    df_discrete = pd.concat(df_data, ignore_index=True)

    startTime_year = []
    for time_entry in df_discrete['Start Time [UTC]']:
        start_time = np.datetime64(dt.datetime.strptime(time_entry,'%Y-%m-%dT%H:%M:%S.%fZ'))
        startTime_year.append(start_time.astype(object).year)

    df_discrete['sampleYear'] = startTime_year

    
    return df_discrete
  


def loadDiscreteData_github():

    # URL on the Github where the csv files are stored
    github_url = 'https://github.com/OOI-CabledArray/discreteSummaries/tree/main/FinalSummaries' 
    gh_baseURL = 'https://raw.githubusercontent.com/OOI-CabledArray/discreteSummaries/refs/heads/main/FinalSummaries/'

    template = pd.read_csv(PARAMS_DIR.joinpath('discreteSummary_template.csv'), dtype=str)
    
    pathKeys_ALL=['Cruise Data','Water Sampling']
    pathKeys_ANY=['Ship Data','Ship_Data','Shipboard Data']

    page = requests.get(github_url).text
    file_names = list(set(re.findall('Cabled-[^"]*_Discrete_Summary\.csv',page)))
    print(file_names)

    df_data = []
    if file_names:
        for file in file_names:
            discrete_URL = gh_baseURL + file
            download = requests.get(discrete_URL)
            if download.status_code == 200:
                df = pd.read_csv(io.StringIO(download.content.decode('utf-8')),na_values=['-9999999'])
                for header in df.keys():  
                    if 'Cruise' in header:
                        print('fix this file header eventually...', header)
                        df.rename(columns={header: 'Cruise'},inplace=True)
                df_data.append(df)
        
    df_discrete = pd.concat(df_data, ignore_index=True)

    startTime_year = []
    for time_entry in df_discrete['Start Time [UTC]']:
        start_time = np.datetime64(dt.datetime.strptime(time_entry,'%Y-%m-%dT%H:%M:%S.%fZ'))
        startTime_year.append(start_time.astype(object).year)
    
    df_discrete['sampleYear'] = startTime_year
        
    
    return df_discrete
     


def extractDiscreteOverlay(site,year,discreteSample_dict,variable):
    all_discrete = loadDiscreteData_github()
    base_site = site.split('-')[0]
    if pd.isnull(discreteSample_dict[variable]['discreteColumn']):
        overlayData_disc = {}
    else:
        overlayData_disc = all_discrete[(all_discrete['sampleYear'] == year) & (all_discrete['Target Asset'].str.contains(base_site,case=False))]
    
    return overlayData_disc
