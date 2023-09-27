import matplotlib.pyplot as plt
import skimage
import argparse

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import partial
from scipy import ndimage

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("../")
sys.path.append("../imagen/")
sys.path.append("../../dataproc")

from imagen_pytorch import Unet, Imagen, ImagenTrainer, NullUnet

import matplotlib
import pickle
import os

from utils import *
from send_emails import *

os.environ['MAGICK_MEMORY_LIMIT'] = str(2**128)
matplotlib.rcParams['animation.embed_limit'] = 2**128

seed_value = 42
torch.manual_seed(seed_value)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)

sys.stdout = open(f'FCAST_LOG_{datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}.log','wt')
print = partial(print, flush=True)
SAVE = True

parser = argparse.ArgumentParser()
parser.add_argument('-region', help='Specify the region name eg. North Indian Ocean')
parser.add_argument('-name', help='Specify the cyclone name eg. Mocha')
parser.add_argument('-start', help='Specify the start timestep (in hrs) eg. 12')
parser.add_argument('-horizon', help='Specify the number of horizon (in days) eg. 2')
args = parser.parse_args()

region = args.region
name = args.name.lower()
start = int(args.start)

print(f"Generating forecast ...\nregion: {region.upper()}\nstart: {start}\nname: {name.upper()}")

cyclone = Cyclone(region, name)
cyclone.load_era5()

ir108_fn = cyclone.metadata['satmaps'][start]['ir108_fn']
ir108_scn = cyclone.get_ir108_data(ir108_fn)    
img = ir108_scn.to_numpy() ; 
img = transform_make_sq_image(img)  
img_o = skimage.transform.resize(img, (64, 64), anti_aliasing=True)

FORECAST_DAYS = int(args.horizon)
horizon = min(cyclone.metadata['count']-start, 
              FORECAST_DAYS*24)

prev_img = img_o
prev_img = torch.from_numpy(prev_img).unsqueeze(0)
img_64_seq = torch.empty(0, 64, 64)
img_64_seq = torch.cat([img_64_seq, prev_img])

era5_64_seq = torch.empty(0, 3, 64, 64)
era5_128_seq = torch.empty(0, 3, 128, 128)

fcdiff_model = FCDiffModel("64_FC_rot904_3e-4", img_o)
srdiff_model = SRDiffModel("64_128_rot904_sep_3e-4", img_o)
tpdiff_model = None

print("Generating forecasts in 64x64 ...")

for satmap_idx in tqdm(range(start, start+horizon)):
    era5_idx = cyclone.metadata['satmaps'][satmap_idx]['era5_idx']
    era5 = cyclone.get_era5_data(era5_idx, gfs=True)
    
    era5_64 = skimage.transform.resize(era5, (3, 64, 64), anti_aliasing=True)
    era5_64 = torch.from_numpy(era5_64)
    era5_64_seq = torch.cat([era5_64_seq, era5_64.unsqueeze(0)])
    
    era5_128 = skimage.transform.resize(era5, (3, 128, 128), anti_aliasing=True)
    era5_128 = torch.from_numpy(era5_128)
    era5_128_seq = torch.cat([era5_128_seq, era5_128.unsqueeze(0)])
    
    if satmap_idx == start: 
        era5_tp = cyclone.get_era5_tp_data(era5_idx)
        era5_tp = skimage.transform.resize(era5_tp, (64, 64), anti_aliasing=True)
        tpdiff_model = TPDiffModel("64_PRP_rot904_3e-4", era5_tp)        
        continue
    
    era5_64 = torch.cat([prev_img, era5_64]).unsqueeze(0)    
    era5_64 = era5_64.reshape(era5_64.shape[0], -1).float()
    
    curr_img = fcdiff_model.get_sampled_image(era5_64)
    curr_img = curr_img.cpu()
    img_64_seq = torch.cat([img_64_seq, curr_img]) 
    prev_img = curr_img

print("Forecast generation completed.")

print("Performing super-resolution to 128x128 ...")
sr_images = srdiff_model.get_sampled_images(img_64_seq, era5_128_seq)
print("Super resolution completed.")

print("Generating 64x64 precipitation maps ...")
tp_images = tpdiff_model.get_sampled_images(img_64_seq, era5_64_seq)
print("Precipitation maps generated.")

actual_era5_tp = torch.empty(0, 64, 64)
actual_ir108 = torch.empty(0, 128, 128)
dates = []

print("Loading actual data ...")

for satmap_idx in tqdm(range(start, start+horizon), disable=True):
    ir108_fn = cyclone.metadata['satmaps'][satmap_idx]['ir108_fn']
    ir108_scn = cyclone.get_ir108_data(ir108_fn)    
    img = ir108_scn.to_numpy() ; 
    img = transform_make_sq_image(img)    
      
    img_n = skimage.transform.resize(img, (128, 128), anti_aliasing=True)
    img_n = torch.from_numpy(img_n).unsqueeze(0)
    actual_ir108 = torch.cat([actual_ir108, img_n])
    
    era5_idx = cyclone.metadata['satmaps'][satmap_idx]['era5_idx']
    era5_tp = cyclone.get_era5_tp_data(era5_idx)
    era5_tp = skimage.transform.resize(era5_tp, (64, 64), anti_aliasing=True)
    era5_tp = torch.from_numpy(era5_tp).unsqueeze(0)
    actual_era5_tp = torch.cat([actual_era5_tp, era5_tp])

    dates.append(cyclone.metadata['satmaps'][satmap_idx]['date'])

tp_images[0] = actual_era5_tp[0]
print("Actual data loaded.")

def update(frame_idx):
    fig.clear()
    axs = fig.subplot_mosaic([['ir', 'tp'], ['ir_pred', 'tp_pred']],
                          gridspec_kw={'width_ratios':[1, 1]})

    get_map_img(m, axs['ir'], 
                img2req(actual_ir108[frame_idx]), 
                cyclone.metadata['map_bounds'])
    axs['ir'].set_title("IR 10.8 µm\nGround Truth")

    get_map_img(m, axs['tp'], 
                actual_era5_tp[frame_idx], 
                cyclone.metadata['map_bounds'], era5=True)
    axs['tp'].set_title("Total Precipitation\nGround Truth")

    get_map_img(m, axs['ir_pred'], 
                img2req(sr_images[frame_idx][0].cpu()), 
                cyclone.metadata['map_bounds'])
    axs['ir_pred'].set_title("IR 10.8 µm\nDiffusion Model Forecast")
    
    if frame_idx != start:
        get_map_img(m, axs['tp_pred'],
                ndimage.minimum_filter(tp_images[frame_idx][0].cpu(), size=3),
                cyclone.metadata['map_bounds'], era5=True)
    else:
        get_map_img(m, axs['tp_pred'],
                    tp_images[frame_idx][0].cpu(),
                    cyclone.metadata['map_bounds'], era5=True)
    axs['tp_pred'].set_title("Total Precipitation\nDiffusion Model Forecast")

    fig.suptitle(f"Cyclone {name.replace('-', ' ').title()}\n{region}\n{dates[frame_idx].strftime('%Y-%m-%d %H:%M')}")

m = Basemap(llcrnrlon=cyclone.metadata['map_bounds'][0], llcrnrlat=cyclone.metadata['map_bounds'][1],
            urcrnrlon=cyclone.metadata['map_bounds'][2], urcrnrlat=cyclone.metadata['map_bounds'][3],
            projection='cyl', resolution='l')

fig = plt.figure(figsize=(8,8), constrained_layout=True)
animation = FuncAnimation(fig, update, frames=horizon, interval=250)

if SAVE: 
    predictions_dict = {
        "actual": {
            "ir_108": actual_ir108,
            "tp": actual_era5_tp,
        },
        "predicted": {
            "ir_108": sr_images,
            "tp": tp_images
        },
        "dates": dates
    }
    with open(f"./pkls/{region_to_abbv[region]}_{start:02}_{name}_forecast.pkl", "wb") as file:
        pickle.dump(predictions_dict, file)
    animation.save(f'./gifs/{region_to_abbv[region]}_{start:02}_{name}_forecast.gif', writer='imagemagick')

plt.close()  

subject = f"[COMPLETED] Forecasts for Cyclone {name.replace('-', ' ').title()}"
message_txt = f"""GIF Processing Completed"""
send_txt_email(message_txt, subject)