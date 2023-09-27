import glob
import satpy
import xarray

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from multiprocessing import Pool, cpu_count
from functools import partial

import traceback
import argparse

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 2**128

import sys
import os
sys.path.append('../dataproc/')
os.environ['MAGICK_MEMORY_LIMIT'] = str(2**128)

from utils import *
from send_emails import *

sys.stdout = open(f'VIZ_LOG_{datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}.log','wt')
print = partial(print, flush=True)

SKIP_FRAMES = 2

abbvs = {
    "nio": "North Indian Ocean",
    "aus": "Australia",
    "wpo": "West Pacific Ocean",
    "wio": "West Indian Ocean",
    "use": "US East",
    "usw": "US West"
}

def get_satmaps(region, name):
    
    ERA5_BASE_DIR = "/vol/bitbucket/pn222/satellite/era5"
    
    if region == "nio":
        map_x0, map_y0 = 78.662109, 20.344627 ; hs_length = 20
        if name == "kyarr":
            map_x0, map_y0 = 65.662109, 20.344627 ; hs_length = 20
        if name == "gulab-shaheen":
            map_x0, map_y0 = 72.662109, 20.344627 ; hs_length = 23
    
    if region == "wpo":
        map_x0, map_y0 = 125.068359, 12.597455 ; hs_length = 25
    
    if region == "aus":
        map_x0, map_y0 = 131.681641, -20.244696 ; hs_length = 25
        if name == "niran":
            map_x0, map_y0 = 149.681641, -20.244696 ; hs_length = 25
    
    if region == "wio":
        map_x0, map_y0 = 46.8691, -18.7669 ; hs_length = 25

    if region == "use":
        map_x0, map_y0 = -80.1918, 25.7617 ; hs_length = 22
        if name == "bonnie":
            map_x0, map_y0 = -100.1918, 2.0617 ; hs_length = 32

    if region == "usw":        
        map_x0, map_y0 = -103.074219, 20.550509 ; hs_length = 10   
        if name == "genevieve":
            map_x0, map_y0 = -106.074219, 15.550509 ; hs_length = 16
        
    map_bounds = get_bbox_square(map_x0, map_y0, hs_length)
    era5_nc_files = sorted(glob.glob(f'{ERA5_BASE_DIR}/data/nc/{name}/*.nc'))
    mc_era5, era5_map_bounds =  get_era5_map(era5_nc_files, map_bounds)

    satmaps = {
        "region": region, 
        "name": name,
        "map_bounds": [float(x) for x in era5_map_bounds],
        "era5_fns": era5_nc_files
    }
    satmaps["satmaps"] = []
    
    if region == "nio":
        IR108_BASE_DIR = "/vol/bitbucket/pn222/satellite/mosdac"
        h5_files = sorted(glob.glob(f"{IR108_BASE_DIR}/data/h5/{name}/*/*.h5"))
        for idx in range(0, len(h5_files), SKIP_FRAMES):
            h5_file = h5_files[idx]
            date = " ".join(h5_file.split('/')[-1].split('_')[1:3])
            date = datetime.strptime(date, "%d%b%Y %H%M")
            date = round_to_closest_hour(date)
            satmaps["satmaps"].append({"date": date, "ir108_fn": h5_file}) 

    if region in ["aus", "wpo"]:
        IR108_BASE_DIR = "/vol/bitbucket/pn222/satellite/himawari"
        hr_dirs = sorted(glob.glob(f"{IR108_BASE_DIR}/data/bz2/{name}/*/*"))
        for idx in range(0, len(hr_dirs), SKIP_FRAMES):
            hr_dir = hr_dirs[idx]
            date = " ".join(hr_dir.split("/")[-2:])
            date = datetime.strptime(date, "%Y-%m-%d %H%M")
            if len(glob.glob(hr_dir+"/*.bz2")) > 0:
                satmaps["satmaps"].append({"date": date, "ir108_fn": hr_dir})  

    if region == "wio":
        IR108_BASE_DIR = "/vol/bitbucket/pn222/satellite/msg"
        nat_files = sorted(glob.glob(f"{IR108_BASE_DIR}/data/native/{name}/*.nat"))
        for idx in range(0, len(nat_files), SKIP_FRAMES):
            nat_file = nat_files[idx]
            date = nat_file.split('/')[-1].split('-')[-2].split('.')[0][:10]
            date = datetime.strptime(date, "%Y%m%d%H")
            satmaps["satmaps"].append({"date": date, "ir108_fn": nat_file})  

    if region == "use":
        IR108_BASE_DIR = "/vol/bitbucket/pn222/satellite/goes_east"
        nc_files = sorted(glob.glob(f"{IR108_BASE_DIR}/data/nc/{name}/*/*.nc"))
        for idx in range(0, len(nc_files), SKIP_FRAMES):
            nc_file = nc_files[idx]
            day = nc_file.split('/')[-2]
            hr = nc_file.split('_')[4][8:10]
            date = " ".join([day, hr])
            date = datetime.strptime(date, "%Y-%m-%d %H")
            satmaps["satmaps"].append({"date": date, "ir108_fn": nc_file})  

    if region == "usw":
        IR108_BASE_DIR = "/vol/bitbucket/pn222/satellite/goes_west"
        nc_files = sorted(glob.glob(f"{IR108_BASE_DIR}/data/nc/{name}/*/*.nc"))
        for idx in range(0, len(nc_files), SKIP_FRAMES):
            nc_file = nc_files[idx]
            day = nc_file.split('/')[-2]
            hr = nc_file.split('_')[4][8:10]
            date = " ".join([day, hr])
            date = datetime.strptime(date, "%Y-%m-%d %H")
            satmaps["satmaps"].append({"date": date, "ir108_fn": nc_file})   
    
    hrs = np.array([np64_to_datetime(x.values) for x in mc_era5["time"]])
    for idx in range(len(satmaps["satmaps"])):
        try:
            era5_idx = np.where(hrs == satmaps["satmaps"][idx]["date"])[0][0]
            satmaps["satmaps"][idx]["era5_idx"] = era5_idx
        except Exception as e:            
            print(f"[{name.upper()}]: Processing error at {satmaps['satmaps'][idx]['date']}")
            print(traceback.format_exc())
            del satmaps["satmaps"][idx]

    satmaps["count"] = len(satmaps["satmaps"])
    satmaps["satmaps"] = sorted(satmaps["satmaps"], key=lambda k: k['era5_idx'])

    return satmaps

GFS_VARIABLES = [
    ("u10", "10m_u_component_of_wind", "10m Wind U Component"),
    ("v10", "10m_v_component_of_wind", "10m Wind V Component"),
    ("tp", "total_precipitation", "Total Precipitation"),
    ("tcc", "total_cloud_cover", "Total Cloud Cover"),
]

def get_map_img(m, ax, data, map_bounds,
                x=None, y=None, contour=False,
                title=None,
                y_labels=[1, 0, 0, 0],
                x_labels=[0, 0, 0, 1], colorbar=False):
    
    kwargs = {
        "linewidth": 0.5,
        "color": "k",
        "ax": ax
    }

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        
    m.drawcoastlines(**kwargs)
    m.drawcountries(**kwargs)
    m.drawstates(**kwargs)

    if contour: im = m.contourf(x, y, data, cmap='jet', ax=ax)
    else: im = m.imshow(data, origin='upper', extent=map_bounds, cmap="gray", ax=ax)

    m.drawparallels(range(int(map_bounds[1])-1, int(map_bounds[3]), 7), labels=y_labels, fontsize=10, ax=ax)
    m.drawmeridians(range(int(map_bounds[0])-1, int(map_bounds[2]), 7), labels=x_labels, fontsize=10, ax=ax)

    if title: ax.set_title(title)

    if colorbar:
        fig.colorbar(im, cax=cax, orientation='vertical')    

    return im

def update(frame_idx, fig, era5, name, satmaps, m, lon_grid, lat_grid):
    fig.clear()
    axs = fig.subplot_mosaic([['main', 'era5_0', 'era5_1'],
                          ['main', 'era5_2', 'era5_3']],
                          gridspec_kw={'width_ratios':[2, 1, 1]})
    ir108_metadata = satmaps['satmaps'][frame_idx]
    if region == "nio":
        ir108_scn = get_insat3d_ir108_scn(ir108_metadata['ir108_fn'], satmaps['map_bounds'])
    if region in ["aus", "wpo"]:
        ir108_scn = get_himawari_ir108_scn(ir108_metadata['ir108_fn'], satmaps['map_bounds'])
    if region == "wio":
        ir108_scn = get_msg_ir108_scn(ir108_metadata['ir108_fn'], satmaps['map_bounds'])
    if region in ["use", "usw"]:
        ir108_scn = get_goes_ir108_scn(ir108_metadata['ir108_fn'], satmaps['map_bounds'])

    get_map_img(m, axs['main'], ir108_scn.data, satmaps['map_bounds'])
    axs['main'].set_title(f"Cyclone {name.replace('-', ' ').title()} - {abbvs[region]}\nIR 10.8 µm\n{ir108_metadata['date'].strftime('%Y-%m-%d %H:%M')}")
    
    for idx, variable in enumerate(GFS_VARIABLES):
        ax = axs[f'era5_{idx}']
        
        x, y = lon_grid, lat_grid
        data = era5.variables[variable[0]][:][ir108_metadata['era5_idx']]
        cf = get_map_img(m, ax, data, satmaps['map_bounds'],
                         lon_grid, lat_grid, 
                         y_labels=[0, 0, 0, 0],
                         x_labels=[0, 0, 0, 0],
                         contour=True)
        
        ax.set_title(f"{variable[2]}")

    ir108_scn.close()

def process_gifs(name):
    print(f"[{name.upper()}] - Processing Cyclone {name.replace('-', ' ').title()}.")
    satmaps = get_satmaps(region, name)
    print(f"[{name.upper()}] - Satmaps created.") 
    
    era5, _ = get_era5_map(satmaps['era5_fns'], map_bounds=satmaps['map_bounds'])
    m = Basemap(llcrnrlon=satmaps['map_bounds'][0], llcrnrlat=satmaps['map_bounds'][1],
                urcrnrlon=satmaps['map_bounds'][2], urcrnrlat=satmaps['map_bounds'][3],
                projection='cyl', resolution='l')
    
    lat = era5.variables["latitude"][:]
    lon = era5.variables["longitude"][:]
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    fig = plt.figure(figsize=(11.15,6), constrained_layout=True)
    update_fn = partial(update, 
                        fig=fig, 
                        era5=era5, 
                        satmaps=satmaps, 
                        m=m,
                        name=name,
                        lon_grid=lon_grid,
                        lat_grid=lat_grid)

    animation = FuncAnimation(fig, update_fn, frames=satmaps['count'], interval=250)
    animation.save(f'./gifs/{region}/{name}_ir108_era5.gif', writer='imagemagick')
    
    plt.close() ; era5.close()
    print(f"[{name.upper()}] - GIF processing completed.")      

    subject = f"[COMPLETED] Processed - Cyclone {name.replace('-', ' ').title()}"
    message_txt = f"""GIF Processing Completed"""
    send_txt_email(message_txt, subject)


parser = argparse.ArgumentParser()
parser.add_argument('-region', help='Specify the region name')
args = parser.parse_args()
region = args.region
    
if region == "nio":  
    # names = sorted([x.split('/')[-1] for x in glob.glob("/vol/bitbucket/pn222/satellite/mosdac/data/h5/*")])
    names = ["gulab-shaheen"]
if region == "aus":
    names = ["ilsa", "seroja", "niran", "damien", "ferdinand", "veronica"]
if region == "wio":
    names = sorted([x.split('/')[-1] for x in glob.glob("/vol/bitbucket/pn222/satellite/msg/data/native/*")])
if region == "use":
    names = sorted([x.split('/')[-1] for x in glob.glob("/vol/bitbucket/pn222/satellite/goes_east/data/nc/*")])
if region == "usw":
    names = sorted([x.split('/')[-1] for x in glob.glob("/vol/bitbucket/pn222/satellite/goes_west/data/nc/*")])
if region == "wpo":
    names = ["noru", "vamco", "rai", "molave", "nesat", "nanmadol",  "chanthu", "goni", "in-fa"]

pool = Pool(cpu_count())
process_func = partial(process_gifs)
results = pool.map(process_func, names)
pool.close()
pool.join()