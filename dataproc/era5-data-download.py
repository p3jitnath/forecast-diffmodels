import datetime
import shutil
import requests
import glob
import os

import zipfile
import glob
import warnings
import matplotlib.pyplot as plt
import pandas as pd

from multiprocessing import Pool, cpu_count
from functools import partial
from send_emails import send_txt_email

import subprocess
import cdsapi

warnings.filterwarnings("ignore")

import sys
sys.stdout = open(f'ERA5_LOG_{datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}.log','wt')
print = partial(print, flush=True)

BASE_DIR = "/vol/bitbucket/pn222/satellite/era5"

ERA5_VARIABLES = [
            '100m_u_component_of_wind', '100m_v_component_of_wind', '10m_u_component_of_neutral_wind',
            '10m_u_component_of_wind', '10m_v_component_of_neutral_wind', '10m_v_component_of_wind',
            '10m_wind_gust_since_previous_post_processing', 'cloud_base_height', 'convective_precipitation',
            'convective_rain_rate', 'high_cloud_cover', 'instantaneous_10m_wind_gust',
            'instantaneous_large_scale_surface_precipitation_fraction', 'large_scale_precipitation', 'large_scale_precipitation_fraction',
            'large_scale_rain_rate', 'low_cloud_cover', 'maximum_total_precipitation_rate_since_previous_post_processing',
            'medium_cloud_cover', 'minimum_total_precipitation_rate_since_previous_post_processing', 'precipitation_type',
            'total_cloud_cover', 'total_column_cloud_ice_water', 'total_column_cloud_liquid_water',
            'total_column_rain_water', 'total_precipitation', 'vertical_integral_of_divergence_of_cloud_frozen_water_flux',
            'vertical_integral_of_divergence_of_cloud_liquid_water_flux', 'vertical_integral_of_eastward_cloud_frozen_water_flux', 'vertical_integral_of_eastward_cloud_liquid_water_flux',
            'vertical_integral_of_northward_cloud_frozen_water_flux', 'vertical_integral_of_northward_cloud_liquid_water_flux',
        ] 

cyclones_path = "./list_of_cyclones.xlsx"
df = pd.read_excel(cyclones_path)
df = df.drop('Unnamed: 8', axis=1)
df = df.dropna()
df = df[df["Name"] == "Genevieve"]

def is_stub_already_present(dest_folder, stub):
  stubs = [x.split('/')[-1] for x in glob.glob(dest_folder+"*.nc")]
  if stub in stubs: 
      print(f"Present: {stub}")
      return True
  return False

def fetch_era5_data(date, nbox, local_path):
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': ERA5_VARIABLES,
            'year': f'{date.year}',
            'month': f'{date.month:02}',
            'day': [f'{date.day:02}'],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': nbox,
        },
        local_path)

def download_era5(date, name, nbox, abbv):
    dest_folder = f"{BASE_DIR}/data/nc/{name.replace(' ', '').lower()}/"
    os.makedirs(dest_folder, exist_ok=True)
    
    stub = f'ERA5_{abbv.upper()}_{date.year}{date.month:02}{date.day:02}.nc'
    local_path = f'{dest_folder}{stub}'

    print(f'[{name}] - {date.strftime("%Y-%m-%d")} - Downloading file ... ')
    try:
        if not is_stub_already_present(dest_folder, stub):
            fetch_era5_data(date, nbox, local_path)
        print(f'[{name}] - {date.strftime("%Y-%m-%d")} - Downloaded.')
    except Exception as e:
        subprocess.run(f"rm -rf {local_path}", shell=True)
        print(f'[{name}] - {date.strftime("%Y-%m-%d")} - Error: {e}') 

INSAT_WBOX = (0.7790000370005146, -81.22400385793298, 163.220007752534, 81.22400385793298)
MSG_WBOX = (-35.69845598820453, -81.2611618767099, 126.69845598820453, 81.2611618767099)
GOES_EAST_WBOX = (-156.19630215232982, -81.14754058985199, 6.196302152329821, 81.14754058985199)
GOES_WEST_WBOX = (-179.9999996295335, -81.14754058985199, 179.99999676657748, 81.14754058985199)
HIMAWARI_WBOX = (-179.99999423788685, -81.05107238178346, 179.9999918966511, 81.05107238178346)

def get_nbox_from_wbox(wbox):
    west_lon, south_lat, east_lon, north_lat = wbox
    nbox = [north_lat, west_lon, south_lat, east_lon]
    return nbox

EGEO_WBOX = (0, -90, 180, 90)
WGEO_WBOX = (-180, -90, 0, 90)

bboxes = {
    'North Indian Ocean': (get_nbox_from_wbox(INSAT_WBOX), "NIO"),
    'Australia': (get_nbox_from_wbox(EGEO_WBOX), "AUS"),
    'South West Indian Ocean': (get_nbox_from_wbox(MSG_WBOX), "SWIO"),
    'North Atlantic Ocean': (get_nbox_from_wbox(GOES_EAST_WBOX), "USE"),
    'North Pacific Ocean': (get_nbox_from_wbox(WGEO_WBOX), "USW"),
    'West Pacific Ocean': (get_nbox_from_wbox(EGEO_WBOX), "PHI")
}

for idx in range(len(df)):
    row = df.iloc[idx]       
    name = row["Name"]
    
    start_date = datetime.datetime.strptime(row["Form Date"], "%d-%m-%Y")
    end_date = datetime.datetime.strptime(row["Dissipated Date"], "%d-%m-%Y")
    nbox, abbv = bboxes[row["Region"]]
    
    current_date = start_date
    dates = [start_date]
    while current_date < end_date:
        current_date += datetime.timedelta(days=1)
        dates.append(current_date)
    
    pool = Pool(cpu_count())
    download_func = partial(download_era5, name=name, nbox=nbox, abbv=abbv)
    results = pool.map(download_func, dates)
    pool.close()
    pool.join()
    
    print(f'[{name}] - All downloads are finished.')
        
    with open("ERA5_COMPLETE.txt", "a+") as file:
      file.write(f"{name}\t{datetime.datetime.now()}\n")
    
    subject = f"[COMPLETED] Download - Cyclone {name}"
    message_txt = f"""Download Completed"""
    send_txt_email(message_txt, subject)
