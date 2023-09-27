import datetime
import zipfile
import glob
import os

from multiprocessing import Pool, cpu_count
from functools import partial
from send_emails import send_txt_email

import sys
sys.stdout = open(f'EU_EXTRACT_LOG_{datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}.log','wt')

name = "*"
extracted_paths = glob.glob(f"/vol/bitbucket/pn222/satellite/msg/data/native/{name}/*.nat")

def extract_files_with_extension(zip_file_path, target_extension, extraction_path):
    name = zip_file_path.split("/")[-2]
    extraction_path += f"{name}/"
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.filename.endswith(target_extension):
                    target_file_path = os.path.join(extraction_path, file_info.filename)
                    if target_file_path in extracted_paths: return
                    zip_ref.extract(file_info.filename, path=extraction_path)
                    print(f"Extracted: {target_file_path}")
    except Exception as e:
        print(f"Error: {zip_file_path} - {e}")

zip_files = glob.glob(f"/vol/bitbucket/pn222/satellite/msg/data/zip/{name}/*.zip")
pool = Pool(cpu_count())
zip_func = partial(extract_files_with_extension, 
                   target_extension = '.nat', 
                   extraction_path = '/vol/bitbucket/pn222/satellite/msg/data/native/')
results = pool.map(zip_func, zip_files)
pool.close()
pool.join()

if name == "*":
    subject = f"[COMPLETED] Extraction Completed for EU Cyclones"
else:
    subject = f"[COMPLETED] Extraction Completed for Cyclone {name.capitalize()}"
message_txt = f"""Extraction Completed"""
send_txt_email(message_txt, subject)
