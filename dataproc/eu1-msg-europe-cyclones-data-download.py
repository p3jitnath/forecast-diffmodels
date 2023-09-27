# METEOSAT Satellite Data Download and Visualisation
# Ref. Notebook: https://gitlab.eumetsat.int/eumetlab/data-services/eumdac_data_store/-/blob/master/3_Downloading_products.ipynb

import eumdac
import datetime
import shutil
import requests
import glob
import os

import zipfile
import warnings
import matplotlib.pyplot as plt
import pandas as pd

from multiprocessing import Pool, cpu_count
from functools import partial
from send_emails import send_txt_email

import time
import sys

warnings.filterwarnings("ignore")

sys.stdout = open(f'EU_LOG_{datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}.log','wt')

BASE_DIR = "/vol/bitbucket/pn222/satellite/msg"

cyclones_path = "./list_of_cyclones.xlsx"
df = pd.read_excel(cyclones_path)
df = df.drop('Unnamed: 8', axis=1)
msg_df = df[df["Satellite Data"] == "EUMETSAT - Meteosat 9"]

consumer_key = INSERT_CONSUMER_KEY
consumer_secret = INSERT_CUSTOMER_SECRET

credentials = (consumer_key, consumer_secret)
token = eumdac.AccessToken(credentials)

try:
    print(f"This token '{token}' expires {token.expiration}")
except requests.exceptions.HTTPError as error:
    print(f"Error when trying the request to the server: '{error}'")

datastore = eumdac.DataStore(token)

try:    
    selected_collection = datastore.get_collection('EO:EUM:DAT:MSG:HRSEVIRI-IODC')
    print(f"{selected_collection} - {selected_collection.title}")
except eumdac.datastore.DataStoreError as error:
    print(f"Error related to the data store: '{error.msg}'")
except eumdac.collection.CollectionError as error:
    print(f"Error related to the collection: '{error.msg}'")
except requests.exceptions.RequestException as error:
    print(f"Unexpected error: {error}")

def filter_hourly_products(eumdac_products):
    eumdac_products = sorted(eumdac_products)
    hourly_products = []
    prev_time = None
    
    for product in eumdac_products:
        product_time = str(product).split("-")[-2][:10]
        if product_time == prev_time:
            continue
        else:
            hourly_products.append(product)
            prev_time = product_time 
    
    return hourly_products

def exclude_already_downloaded(eumdac_products, name):
    downloaded_files = glob.glob(f"{BASE_DIR}/data/zip/{name.lower()}/*.zip")
    download_file_times = [x.split("/")[-1][:-4] for x in downloaded_files]
    remaining_products = []
    for product in eumdac_products:
        if str(product) in download_file_times:
            continue
        else:
            remaining_products.append(product)
    return remaining_products

def get_eumdac_products(start, end, hourly=True):
    eumdac_products = selected_collection.search(
      dtstart=start,
      dtend=end)
    
    if hourly: eumdac_products = filter_hourly_products(eumdac_products)
    eumdac_products = exclude_already_downloaded(eumdac_products, name)
    return eumdac_products

def fetch_product_from_server(product, name):
    try:
        with product.open() as fsrc, \
                open(f"{BASE_DIR}/data/zip/{name.lower()}/{fsrc.name}", mode='wb') as fdst:
            print(f'Download of product {product} started ...')
            shutil.copyfileobj(fsrc, fdst)
            print(f'Download of product {product} finished.')
            time.sleep(5)
    except eumdac.product.ProductError as error:
        print(f"Error related to the product '{product}' while trying to download it: '{error.msg}'")
    except requests.exceptions.RequestException as error:
        print(f"Unexpected error: {error}")

def download_products(products, name):
    os.makedirs(f"{BASE_DIR}/data/zip/{name.lower()}", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/data/native/{name.lower()}", exist_ok=True)
    
    pool = Pool(cpu_count())
    download_func = partial(fetch_product_from_server, name=name)
    results = pool.map(download_func, products)
    pool.close()
    pool.join()
      
    print('All downloads are finished.')   
    
    if len(products) > 0:
        with open("EU_COMPLETE.txt", "a+") as file:
           file.write(f"{name}\t{datetime.datetime.now()}\n")
        subject = f"[COMPLETED] Download - Cyclone {name}"
        message_txt = f"""Download Completed"""
        send_txt_email(message_txt, subject)

for idx in range(len(msg_df)):
    row = msg_df.iloc[idx]
    name = row['Name']
    start_date = datetime.datetime.strptime(row["Form Date"], "%d-%m-%Y")
    end_date = datetime.datetime.strptime(row["Dissipated Date"], "%d-%m-%Y") + datetime.timedelta(days=1)
    
    eumdac_products = get_eumdac_products(start_date, end_date, name)
    print(f"Cyclone [{name}]\tTotal number of products:\t{len(eumdac_products)}")
    print(f"Downloading Cyclone {name} ...")
    download_products(eumdac_products, name)
    print("--------")