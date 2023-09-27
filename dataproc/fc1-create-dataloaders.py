import pandas as pd
import torch
import skimage
import pickle

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

from utils import *
from send_emails import send_txt_email

BASE_DIR = "/vol/bitbucket/pn222/satellite/dataloader/64_FC/"

import sys
sys.stdout = open(f'DL_FC_LOG_{datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}.log','wt')

cyclones_path = "./list_of_cyclones.xlsx"
df = pd.read_excel(cyclones_path)
df = df.drop('Unnamed: 8', axis=1)
df = df.dropna()

def is_stub_already_present(dest_folder, stub):
    stubs = [x.split('/')[-1] for x in glob.glob(dest_folder+"*.dat")]
    if stub in stubs: 
        print(f"Present: {stub}", flush=True)
        return True
    return False

def fetch_cyclone(idx):       
    row = df.iloc[idx]
    region = row["Region"]
    name = row["Name"]
    cyclone = Cyclone(region, name)
    cyclone.load_era5()
    
    o_size = 64 ; n_size = 128
    data_loader = CycloneDataLoader(mode="fc")
    
    region = region_to_abbv[region]
    name = name.replace(' ', '').lower()
    filename = f"{region}_{name}.dat"
    
    if is_stub_already_present(BASE_DIR, filename):
        return
    
    print(f"[{name.upper()}] Processing dataloader.", flush=True)
    
    for satmap_idx in tqdm(range(cyclone.metadata['count']), disable=True):
        if satmap_idx == 0: 
            continue

        old_satmap_idx = satmap_idx-1
        cur_satmap_idx = satmap_idx
        
        ir108_fn = cyclone.metadata['satmaps'][cur_satmap_idx]['ir108_fn']
        ir108_scn = cyclone.get_ir108_data(ir108_fn)    
        img = ir108_scn.to_numpy() ; 
        img = transform_make_sq_image(img)    
          
        img_o = skimage.transform.resize(img, (o_size, o_size), anti_aliasing=True)
        img_o = torch.from_numpy(img_o).unsqueeze(0)        
        img_n = torch.zeros((1, n_size, n_size))
        
        era5_idx = cyclone.metadata['satmaps'][cur_satmap_idx]['era5_idx']
        era5 = cyclone.get_era5_data(era5_idx, gfs=True)
        era5 = skimage.transform.resize(era5, (3, o_size, o_size), anti_aliasing=True)
        era5 = torch.from_numpy(era5)

        ir108_fn = cyclone.metadata['satmaps'][old_satmap_idx]['ir108_fn']
        ir108_scn = cyclone.get_ir108_data(ir108_fn)    
        img = ir108_scn.to_numpy() ; 
        img = transform_make_sq_image(img) 
        img = skimage.transform.resize(img, (o_size, o_size), anti_aliasing=True)
        img = torch.from_numpy(img).unsqueeze(0)
        
        era5 = torch.cat([img, era5]).unsqueeze(0)
               
        if torch.isnan(img_o.sum()) or torch.isnan(img_n.sum()) or torch.isnan(era5.sum()):
            print(f"[NAN]\t{region}\t{name}\t{satmap_idx}", flush=True)
            continue
        
        data_loader.add_image(img_o, img_n, era5)
    
    with open(f'{BASE_DIR}{filename}', 'wb') as data_file:
        pickle.dump(data_loader, data_file)
    
    print(f"[{name.upper()}] Completed processing dataloader.", flush=True)

    subject = f"[COMPLETED] Dataloader Processed - Cyclone {name.replace('-', ' ').title()}"
    message_txt = f"""Processing Completed"""
    send_txt_email(message_txt, subject)

from multiprocessing import Pool, cpu_count
from functools import partial

idx = list(range(len(df)))

pool = Pool(cpu_count())
fetch_cyclone_func = partial(fetch_cyclone)
results = pool.map(fetch_cyclone_func, idx)
pool.close()
pool.join()