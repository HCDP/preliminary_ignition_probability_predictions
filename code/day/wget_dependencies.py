import os
import sys
import subprocess
import pytz
import requests
import pandas as pd
from datetime import datetime, timedelta

#paste into env 'https://ikeauth.its.hawaii.edu/files/v2/download/public/system/ikewai-annotated-data/HCDP/'
public_url = os.environ.get('IKEWAI_BASE')
local_dep_dir = os.environ.get('DEPENDENCY_DIR')
hcdp_api_token = os.environ.get('HCDP_API_TOKEN')
county_list = ['bi','ka','mn','oa']

if len(sys.argv) > 1:
    #Set input to specific date
    target_date = sys.argv[1]
    targ_dt = pd.to_datetime(target_date)
else:
    #Set to previous day
    hst = pytz.timezone('HST')
    today = datetime.today().astimezone(hst)
    targ_dt = today - timedelta(days = 1)


for county in county_list:
    #import statics (model training)
    #-- land cover
    lc_url = public_url + 'workflow_data/preliminary/ignition_prob/dependencies/' + 'PerCov2016model_'+county+'.tif'
    local_name = local_dep_dir + 'land_cover_'+county+'.tif'
    cmd = ["wget",lc_url,"-O",local_name]
    lc_rtn_code = subprocess.run(cmd).returncode
    #-- training dataset
    tr_url = public_url + 'workflow_data/preliminary/ignition_prob/dependencies/' + 'FireData-2002-2019_'+county+'.csv'
    local_name = local_dep_dir + 'firedata_'+county+'.csv'
    cmd = ["wget",tr_url,"-O",local_name]
    tr_rtn_code = subprocess.run(cmd).returncode
    #-- reference geotiff
    ref_url = public_url + 'workflow_data/preliminary/ignition_prob/dependencies/' + 'ref_'+county+'.tif'
    local_name = local_dep_dir + 'ref_'+county+'.tif'
    cmd = ["wget",ref_url,"-O",local_name]
    ref_rtn_code = subprocess.run(cmd).returncode #check that this executes cmd correctly

    #import daily updated data
    #--ndvi
    date_fmt = targ_dt.strftime('%Y-%m-%d')
    finish_flag = 0
    attempts = 0
    print(f'Getting NDVI for {date_fmt}')
    while (finish_flag < 1)&(attempts<10):
        ndvi_url = f"https://api.hcdp.ikewai.org/raster?date={date_fmt}&extent={county}&datatype=ndvi_modis&period=day&returnEmptyNotFound=False"
        ndvi_file = local_dep_dir + 'NDVI_'+county+'.tif'
        try:
            req = requests.get(ndvi_url,headers={'Authorization':f'Bearer {hcdp_api_token}'})
            req.raise_for_status()
            with open(ndvi_file,'wb') as f:
                f.write(req.content)
            finish_flag = 1
            attempts += 1
        except requests.exceptions.HTTPError as err:
            print(f'NDVI for {date_fmt} not found. Fetching previous day.')
            dt = pd.to_datetime(date_fmt) - pd.Timedelta(days=1)
            date_fmt = dt.strftime('%Y-%m-%d')

    #--precip (Preciptation.tif)
    #Reset loop flag and date
    date_fmt = targ_dt.strftime('%Y-%m-%d')
    finish_flag = 0
    attempts = 0
    print(f'Getting precip for {date_fmt}')
    while (finish_flag < 1)&(attempts<10):
        prec_url = f"https://api.hcdp.ikewai.org/raster?date={date_fmt}&extent={county}&datatype=rainfall&production=new&period=day&returnEmptyNotFound=False"
        prec_file = local_dep_dir + 'Preciptation_'+county+'.tif'
        try:
            req = requests.get(prec_url,headers={'Authorization':f'Bearer {hcdp_api_token}'})
            req.raise_for_status()
            with open(prec_file,'wb') as f:
                f.write(req.content)
            finish_flag = 1
            attempts += 1
        except requests.exceptions.HTTPError as err:
            print(f'Precipitation for {date_fmt} not found. Fetching previous day.')
            dt = pd.to_datetime(date_fmt) - pd.Timedelta(days=1)
            date_fmt = dt.strftime('%Y-%m-%d')
            
    #--max temp (Tmax.tif)
    #Reset loop flag and date and attempts
    date_fmt = targ_dt.strftime('%Y-%m-%d')
    finish_flag = 0
    attempts = 0
    print(f'Getting Tmax for {date_fmt}')
    while (finish_flag < 1)&(attempts<10):
        tmax_url = f"https://api.hcdp.ikewai.org/raster?date={date_fmt}&extent={county}&datatype=temperature&aggregation=max&period=day&returnEmptyNotFound=False"
        tmax_file = local_dep_dir + 'Tmax_'+county+'.tif'
        try:
            req = requests.get(tmax_url,headers={'Authorization':f'Bearer {hcdp_api_token}'})
            req.raise_for_status()
            with open(tmax_file,'wb') as f:
                f.write(req.content)
            finish_flag = 1
            attempts += 1
        except requests.exceptions.HTTPError as err:
            print(f'Tmax for {date_fmt} not found. Fetching previous day.')
            dt = pd.to_datetime(date_fmt) - pd.Timedelta(days=1)
            date_fmt = dt.strftime('%Y-%m-%d')

    #--relhum (RH.tif)
    #Reset loop flag and date and attempts
    date_fmt = targ_dt.strftime('%Y-%m-%d')
    finish_flag = 0
    attempts = 0
    print(f'Getting RH for {date_fmt}')
    while (finish_flag < 1)&(attempts<10):
        rh_url = f"https://api.hcdp.ikewai.org/raster?date={date_fmt}&extent={county}&datatype=relative_humidity&period=day&returnEmptyNotFound=False"
        rh_file = local_dep_dir + 'RH_'+county+'.tif'
        try:
            req = requests.get(rh_url,headers={'Authorization':f'Bearer {hcdp_api_token}'})
            req.raise_for_status()
            with open(rh_file,'wb') as f:
                f.write(req.content)
            finish_flag = 1
            attempts += 1
        except requests.exceptions.HTTPError as err:
            print(f'RH for {date_fmt} not found. Fetching previous day.')
            dt = pd.to_datetime(date_fmt) - pd.Timedelta(days=1)
            date_fmt = dt.strftime('%Y-%m-%d')