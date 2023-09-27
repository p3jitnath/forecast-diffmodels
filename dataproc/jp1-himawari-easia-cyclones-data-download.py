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
import sys

warnings.filterwarnings("ignore")

sys.stdout = open(f'JP_LOG_{datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}.log','wt')

BASE_DIR = "/vol/bitbucket/pn222/satellite/himawari"

cyclones_path = "./list_of_cyclones.xlsx"
df = pd.read_excel(cyclones_path)
df = df.drop('Unnamed: 8', axis=1)
himawari_df = df[df["Name"] == "Ilsa"]

def is_stub_already_present(dest_folder, stub):
    stubs = [x.split('/')[-1] for x in glob.glob(dest_folder+"*.bz2")]
    if stub in stubs: 
        print(f"Present: {stub}")
        return True
    return False

def fetch_aws_file(year, month, day, hour, stub, dest_folder):
    statement = f"aws s3 cp --no-sign-request s3://noaa-himawari9/AHI-L1b-FLDK/{year}/{month:02}/{day:02}/{hour:02}00/{stub} {dest_folder}"
    command = statement.split()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output, error = process.communicate()

def download_himawara_b13(date, name): 
    year = date.year ; month = date.month ; day = date.day ; hour = date.hour
    statement = f"aws s3 ls --no-sign-request s3://noaa-himawari9/AHI-L1b-FLDK/{year}/{month:02}/{day:02}/{hour:02}00/HS_H09_{year}{month:02}{day:02}_{hour:02}00_B13_FLDK_"
    command = statement.split()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output, error = process.communicate()
    filenames = [x.split(' ')[-1] for x in output.split('\n') if x != '']
    
    dest_folder = f"{BASE_DIR}/data/bz2/{name.lower()}/{year}-{month:02}-{day:02}/{hour:02}00/"
    os.makedirs(dest_folder, exist_ok=True)
    
    print(f'[{name}] - {date.strftime("%Y-%m-%d %H:%M")} - Downloading files ... ')
    for stub in filenames:
        if not is_stub_already_present(dest_folder, stub):
          fetch_aws_file(year, month, day, hour, stub, dest_folder)
    print(f'[{name}] - {date.strftime("%Y-%m-%d %H:%M")} - Downloaded.')

for idx in range(len(himawari_df)):
    row = himawari_df.iloc[idx]
    name = row["Name"]
    start_date = datetime.datetime.strptime(row["Form Date"], "%d-%m-%Y")
    end_date = datetime.datetime.strptime(row["Dissipated Date"], "%d-%m-%Y") + datetime.timedelta(days=1)
    
    current_date = start_date
    dates = [start_date]
    while current_date < end_date:
        current_date += datetime.timedelta(hours=1)
        dates.append(current_date)
        
    pool = Pool(cpu_count())
    download_func = partial(download_himawara_b13, name=name)
    results = pool.map(download_func, dates)
    pool.close()
    pool.join()
    
    print(f'[{name}] - All downloads are finished.')
        
    with open("JP_COMPLETE.txt", "a+") as file:
        file.write(f"{name}\t{datetime.datetime.now()}\n")
    
    subject = f"[COMPLETED] Download - Cyclone {name}"
    message_txt = f"""Download Completed"""
    send_txt_email(message_txt, subject)