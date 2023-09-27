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

import paramiko
import socket
import subprocess
import sys

import time

warnings.filterwarnings("ignore")

sys.stdout = open(f'IN_LOG_{datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}.log','wt')

BASE_DIR = "/vol/bitbucket/pn222/satellite/mosdac"

cyclones_path = "./list_of_cyclones.xlsx"
df = pd.read_excel(cyclones_path)
df = df.drop('Unnamed: 8', axis=1)
insat_df = df[df["Satellite Data"] == "ISRO - INSAT"]

def fetch_sftp_connection(hostname, port, username, password):
    sock = socket.socket(socket.AF_INET)
    sock.connect((hostname, port))
    transport = paramiko.Transport(sock)
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    return sftp, transport

def close_sftp_connection(sftp, transport):
    sftp.close()
    transport.close()

def download_file_sftp(remote_path, local_path):
    sftp, transport = fetch_sftp_connection(hostname, port, username, password)
    sftp.get(remote_path, local_path)
    close_sftp_connection(sftp, transport)
    time.sleep(5)

def is_stub_already_present(dest_folder, stub):
    stubs = [x.split('/')[-1] for x in glob.glob(dest_folder+"*.h5")]
    if stub in stubs: 
        print(f"Present: {stub}")
        return True
    return False

def download_insat3d(date, order_no, name, count=1):
    year = date.year ; month = date.month ; day = date.day ; hour = date.hour ;
    
    stub = "3DIMG_" + f"{date.strftime('%d%b%Y_%H%M').upper()}" + "_L1B_STD_V01R00.h5"
    remote_path = f"Order/{order_no}/{stub}"
    dest_folder = f"{BASE_DIR}/data/h5/{name.replace(' ', '').lower()}/{year}-{month:02}-{day:02}/" ; local_path  = f"{dest_folder}{stub}"
    os.makedirs(dest_folder, exist_ok=True)
    
    print(f'[{name}] - {date.strftime("%Y-%m-%d %H:%M")} - Downloading file ... ')
    try:
        if not is_stub_already_present(dest_folder, stub):
            download_file_sftp(remote_path, local_path)
        print(f'[{name}] - {date.strftime("%Y-%m-%d %H:%M")} - Downloaded.')
    except Exception as e:
        subprocess.run(f"rm -rf {local_path}", shell=True)
        print(f'[{name}] - {date.strftime("%Y-%m-%d %H:%M")} - Error: {e}')
        if count < 2: return download_insat3d(date + datetime.timedelta(minutes=-1), order_no, name, count=count+1)
        

ORDER_NO_MAPPINGS = {
    "Fani"  : "Jun23_086424",
    "Bulbul": "Jun23_087006",
    "Mocha" : "Jun23_086446",
    "Asani" : "Jun23_086448",
    "Yaas"  : "Jun23_086450",
    "Nivar" : "Jun23_087004",
    "Amphan": "Jun23_086454",
    "Gulab - Shaheen"  : "Jun23_086472",
    "Tauktae" : "Jun23_086474",
    "Nisarga" : "Jun23_086476",
    "Maha"    : "Jun23_086478",
    "Kyarr"   : "Jun23_086480",
    "Vayu"    : "Jun23_086482"
}

hostname = 'download.mosdac.gov.in'
port = 22
username = 'ukp3jitnath'
password = 'Mosdac2@23'

for name in ORDER_NO_MAPPINGS.keys():    
    row = df.loc[df["Name"] == name].squeeze()
    order_no = ORDER_NO_MAPPINGS[name]
    start_date = datetime.datetime.strptime(row["Form Date"], "%d-%m-%Y")
    end_date = datetime.datetime.strptime(row["Dissipated Date"], "%d-%m-%Y") + datetime.timedelta(hours=23)
    
    current_date = start_date
    dates = [start_date]
    while current_date < end_date:
        current_date += datetime.timedelta(hours=1)
        dates.append(current_date)
    
    pool = Pool(cpu_count())
    download_func = partial(download_insat3d, order_no=order_no, name=name)
    results = pool.map(download_func, dates)
    pool.close()
    pool.join()
    
    print(f'[{name}] - All downloads are finished.')
          
    with open("IN_COMPLETE.txt", "a+") as file:
        file.write(f"{name}\t{datetime.datetime.now()}\n")
    
    subject = f"[COMPLETED] Download - Cyclone {name}"
    message_txt = f"""Download Completed"""
    send_txt_email(message_txt, subject)
