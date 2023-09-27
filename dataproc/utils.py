import glob
import pickle
import xarray
import satpy
import torch
import fsspec
import numpy as np

from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta
from pyproj import Proj

import sys
sys.path.append("/homes/pn222/Work/MSc_Project_pn222/imagen/imagen/")

from imagen_pytorch import Unet, Imagen, ImagenTrainer, NullUnet

abbv_to_region = {
    "nio": "North Indian Ocean",
    "aus": "Australia",
    "wpo": "West Pacific Ocean",
    "wio": "West Indian Ocean",
    "use": "North Atlantic Ocean",
    "usw": "North Pacific Ocean"
}

region_to_abbv = {
    "North Indian Ocean": "nio",
    "Australia": "aus",
    "West Pacific Ocean": "wpo",
    "West Indian Ocean": "wio",
    "North Atlantic Ocean": "use",
    "North Pacific Ocean": "usw"
}

GFS_VARIABLES = [
    ("u10", "10m_u_component_of_wind", "10m Wind U Component"),
    ("v10", "10m_v_component_of_wind", "10m Wind V Component"),
    ("tcc", "total_cloud_cover", "Total Cloud Cover"),
    ("tp", "total_precipitation", "Total Precipitation")
]


# ---------------------------------------------
# Classes defined for Model Loading
# ---------------------------------------------

class FCDiffModel:
    def __init__(self, run_name, img_o, woERA5=False):
        self.run_name = run_name
        self.img_o = img_o
        self.max_value, self.min_value = self.img_o.max(), self.img_o.min()
        self.woERA5 = woERA5
        self._init_diff_model()        
        
    def _init_diff_model(self):
        best_epoch_dict = {
            "64_FC_rot904_sep_3e-4": 180,
            "64_FC_rot904_3e-4": 240,
            "64_FC_3e-4": 235,
            "64_FC_woERA5_rot904_3e-4": 220
        }
        
        unet1 = Unet(
            dim = 32,
            cond_dim = 1024,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = 3,
            layer_attns = (False, True, True, True),
        )  
        
        unets = [unet1]
        best_epoch = best_epoch_dict[self.run_name]

        if self.woERA5:
            imagen = Imagen(
                unets = unets,
                image_sizes = (64),
                timesteps = 250,
                cond_drop_prob = 0.1,
                condition_on_continuous = True,
                continuous_embed_dim = 64*64*1,
            )
            ckpt_trainer_path = f"/vol/bitbucket/pn222/models/{self.run_name}/models/64_FC_woERA5/ckpt_trainer_1_{best_epoch:03}.pt"
        else:    
            imagen = Imagen(
                unets = unets,
                image_sizes = (64),
                timesteps = 250,
                cond_drop_prob = 0.1,
                condition_on_continuous = True,
                continuous_embed_dim = 64*64*4,
            )
            ckpt_trainer_path = f"/vol/bitbucket/pn222/models/{self.run_name}/models/64_FC/ckpt_trainer_1_{best_epoch:03}.pt"
        
        trainer = ImagenTrainer(imagen, lr=3e-4, verbose=False).cuda()
        trainer.load(ckpt_trainer_path)  
        self.imagen = imagen

    def get_sampled_image(self, cond_embeds):
        sampled_image = self.imagen.sample(
                batch_size = 1,          
                cond_scale = 3.,
                continuous_embeds=cond_embeds.float().cuda(),
                use_tqdm = False
            )
    
        sampled_image = sampled_image[0, 0, :, :]
        sampled_image = unnormalize(sampled_image, self.max_value, self.min_value)
        return sampled_image.unsqueeze(0)


class SRDiffModel:
    def __init__(self, run_name, img_o, BATCH_LIMIT=32, woERA5=False):
        self.run_name = run_name
        self.img_o = img_o
        self.max_value, self.min_value = self.img_o.max(), self.img_o.min()
        self.BATCH_LIMIT = BATCH_LIMIT
        self.woERA5 = woERA5
        self._init_diff_model()

    def _init_diff_model(self):        
        best_epoch_dict = {
            "64_128_1e-5": 45,
            "64_128_1e-4": 240,
            "64_128_3e-4": 220,    
            "64_128_1k_3e-4": 255,
            "64_128_rot904_3e-4": 85,
            "64_128_sep_3e-4": 220,
            "64_128_rot904_sep_3e-4": 135,
            "64_128_woERA5_rot904_sep_3e-4": 255
        }
                
        unet1 = NullUnet()  
    
        unet2 = Unet(
            dim = 32,
            cond_dim = 1024,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = (2, 4, 8, 8),
            layer_attns = (False, False, False, True),
            layer_cross_attns = (False, False, False, True)
        )
        unets = [unet1, unet2]
        best_epoch = best_epoch_dict[self.run_name]

        if self.woERA5:
            imagen = Imagen(
                unets = unets,
                image_sizes = (64, 128),
                timesteps = 250,
                cond_drop_prob = 0.1,
                condition_on_continuous = False,
                continuous_embed_dim = None,
            )
            ckpt_trainer_path = f"/vol/bitbucket/pn222/models/{self.run_name}/models/64_128_woERA5/ckpt_trainer_2_{best_epoch:03}.pt"
        else:
            imagen = Imagen(
                unets = unets,
                image_sizes = (64, 128),
                timesteps = 250,
                cond_drop_prob = 0.1,
                condition_on_continuous = True,
                continuous_embed_dim = 128*128*3,
            )
            ckpt_trainer_path = f"/vol/bitbucket/pn222/models/{self.run_name}/models/64_128/ckpt_trainer_2_{best_epoch:03}.pt"        
        
        trainer = ImagenTrainer(imagen, lr=3e-4, verbose=False).cuda()
        trainer.load(ckpt_trainer_path)  
    
        self.imagen = imagen

    def get_sampled_images(self, img_64, era5_128=None):
        if not self.woERA5: era5 = era5_128.reshape(era5_128.shape[0], -1)
        img_64 = to3channel(normalize(img_64, 
                                      self.max_value, 
                                      self.min_value))

        sampled_images = torch.empty(0, 3, 128, 128)
        batch_count = int(np.ceil(img_64.shape[0] / self.BATCH_LIMIT))

        for i in tqdm(range(batch_count)):
            img_64_batch  = img_64[self.BATCH_LIMIT*i:self.BATCH_LIMIT*(i+1)]
            if not self.woERA5: 
                era5_128_batch = era5[self.BATCH_LIMIT*i:self.BATCH_LIMIT*(i+1)]
                gen_images = self.imagen.sample(
                        batch_size = img_64_batch.shape[0],
                        start_at_unet_number = 2,              
                        start_image_or_video = img_64_batch.float().cuda(),
                        cond_scale = 3.,
                        continuous_embeds = era5_128_batch.float().cuda(),
                        use_tqdm = False
                )
            else:
                gen_images = self.imagen.sample(
                        batch_size = img_64_batch.shape[0],
                        start_at_unet_number = 2,              
                        start_image_or_video = img_64_batch.float().cuda(),
                        cond_scale = 3.,
                        use_tqdm = False
                )

            gen_images = gen_images.cpu()
            sampled_images = torch.cat([sampled_images, gen_images])
            
        sampled_images = unnormalize(sampled_images, 
                                   self.max_value, 
                                   self.min_value)
    
        return sampled_images


class TPDiffModel:
    def __init__(self, run_name, era5_tp, BATCH_LIMIT=32, woERA5=False):
        self.run_name = run_name
        self.era5_tp = era5_tp
        self.max_value, self.min_value = self.era5_tp.max(), self.era5_tp.min()
        self.BATCH_LIMIT = BATCH_LIMIT
        self.woERA5 = woERA5
        self._init_diff_model()        
        
    def _init_diff_model(self):
        best_epoch_dict = {
            "64_PRP_rot904_3e-4": 260,
            "64_PRP_rot904_sep_3e-4": 210,
            "64_PRP_woERA5_rot904_3e-4": 260
        }
        
        unet1 = Unet(
            dim = 32,
            cond_dim = 1024,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = 3,
            layer_attns = (False, True, True, True),
        )  
        
        unets = [unet1]
        best_epoch = best_epoch_dict[self.run_name]

        if self.woERA5:
            imagen = Imagen(
                unets = unets,
                image_sizes = (64),
                timesteps = 250,
                cond_drop_prob = 0.1,
                condition_on_continuous = True,
                continuous_embed_dim = 64*64*1,
            )
            ckpt_trainer_path = f"/vol/bitbucket/pn222/models/{self.run_name}/models/64_PRP_woERA5/ckpt_trainer_1_{best_epoch:03}.pt"
        else:            
            imagen = Imagen(
                unets = unets,
                image_sizes = (64),
                timesteps = 250,
                cond_drop_prob = 0.1,
                condition_on_continuous = True,
                continuous_embed_dim = 64*64*4,
            )
            ckpt_trainer_path = f"/vol/bitbucket/pn222/models/{self.run_name}/models/64_PRP/ckpt_trainer_1_{best_epoch:03}.pt"        
        
        trainer = ImagenTrainer(imagen, lr=3e-4, verbose=False).cuda()
        trainer.load(ckpt_trainer_path)  
    
        self.imagen = imagen

    def get_sampled_images(self, img_64, era5_64=None):
        if not self.woERA5: img_64 = torch.cat([img_64.unsqueeze(1), era5_64], dim=1)
        else: img_64 = torch.cat([img_64.unsqueeze(1)], dim=1)
            
        cond_embeds = img_64.reshape(img_64.shape[0], -1)
        sampled_images = torch.empty(0, 3, 64, 64)
        batch_count = int(np.ceil(img_64.shape[0] / self.BATCH_LIMIT))
        
        for i in tqdm(range(batch_count)):
            cond_embeds_batch = cond_embeds[self.BATCH_LIMIT*i:self.BATCH_LIMIT*(i+1)]
            gen_images = self.imagen.sample(
                        batch_size = cond_embeds_batch.shape[0],          
                        cond_scale = 3.,
                        continuous_embeds=cond_embeds_batch.float().cuda(),
                        use_tqdm = False
            )
        
            gen_images = gen_images.cpu()
            sampled_images = torch.cat([sampled_images, gen_images])
    
        sampled_images = unnormalize(sampled_images, 
                                   self.max_value, 
                                   self.min_value)
        
        return sampled_images
        
# ---------------------------------------------
# Classes defined for Model Creation Tasks
# ---------------------------------------------

class Cyclone:            
    def _get_filename(self, region, name):
        self.abbv_region = region_to_abbv[region]
        name = name.replace(' ', '').lower()
        return f"{self.abbv_region}_{name}"

    def __init__(self, region, name):        
        self.BASE_DIR = "/vol/bitbucket/pn222/satellite/metadata"
        self.filename = self._get_filename(region, name)
        with open(f"{self.BASE_DIR}/{self.filename}.metadata", 'rb') as metadata_file:
            self.metadata = pickle.load(metadata_file)

    def load_era5(self):
        self.era5, _ = get_era5_map(self.metadata['era5_fns'], map_bounds=self.metadata['map_bounds'])

    def get_era5_tp_data(self, era5_idx):
        key = "tp"
        era5_tp_data = self.era5.variables[key][:][era5_idx].to_numpy()
        return era5_tp_data

    def get_era5_data(self, era5_idx, gfs=True):
        if not gfs:
            era5_keys = list(self.era5.keys())
        else:
            era5_keys = [x[0] for x in GFS_VARIABLES]
        era5_data = []
        for key in era5_keys:
            if (key == "tp"):
                continue
            data = self.era5.variables[key][:][era5_idx].to_numpy()
            era5_data.append(data)
        era5_data = np.array(era5_data)
        return era5_data

    def get_ir108_data(self, ir108_fn):
        region = self.abbv_region
        if region == "nio":
            ir108_scn_fn = get_insat3d_ir108_scn
        if region in ["aus", "wpo"]:
            ir108_scn_fn = get_himawari_ir108_scn
        if region in ["use", "usw"]:
            ir108_scn_fn = get_goes_ir108_scn
        if region == "wio":
            ir108_scn_fn = get_msg_ir108_scn
        ir108_scn = ir108_scn_fn(ir108_fn, self.metadata['map_bounds'])
        return ir108_scn


class CycloneDataLoader:
    def __init__(self, mode="sr"):
        
        if mode == "sr":
            self.img_64  = torch.empty((0, 64, 64), 
                                   dtype=torch.float32)
            self.img_128 = torch.empty((0, 128, 128), 
                                    dtype=torch.float32)
            self.era5 = torch.empty((0, 3, 128, 128), 
                                    dtype=torch.float32)
        if mode == "tp":
            self.img_64  = torch.empty((0, 4, 64, 64), 
                                   dtype=torch.float32)
            self.img_128 = torch.empty((0, 128, 128), 
                                    dtype=torch.float32)            
            self.era5 = torch.empty((0, 64, 64), 
                                    dtype=torch.float32)

        if mode == "fc":
            self.img_64  = torch.empty((0, 64, 64), 
                                   dtype=torch.float32)
            self.img_128 = torch.empty((0, 128, 128), 
                                    dtype=torch.float32)            
            self.era5 = torch.empty((0, 4, 64, 64), 
                                    dtype=torch.float32)

    
    def add_image(self, img_64, img_128, era5):
        self.img_64  = torch.cat((self.img_64, img_64), 0)
        self.img_128 = torch.cat((self.img_128, img_128), 0)
        self.era5 = torch.cat((self.era5, era5), 0)

class ModelDataLoader:
    def __init__(self, batch_size, o_size=64, n_size=128, 
                 augment=False, 
                 test=False, mode="sr", shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle        
        
        self.mode = mode
                     
        if mode == "sr":
            self.img_o = torch.empty((0, o_size, o_size), 
                                   dtype=torch.float32)
            self.img_n = torch.empty((0, n_size, n_size), 
                                    dtype=torch.float32)
            self.era5 = torch.empty((0, 3, n_size, n_size), 
                                    dtype=torch.float32)
        if mode == "tp":
            self.img_o = torch.empty((0, 4, o_size, o_size), 
                                   dtype=torch.float32)
            self.img_n = torch.empty((0, n_size, n_size), 
                                    dtype=torch.float32)
            self.era5 = torch.empty((0, o_size, o_size), 
                                    dtype=torch.float32)

        if mode == "fc":
            self.img_o = torch.empty((0, o_size, o_size), 
                                   dtype=torch.float32)
            self.img_n = torch.empty((0, n_size, n_size), 
                                    dtype=torch.float32)
            self.era5 = torch.empty((0, 4, o_size, o_size), 
                                    dtype=torch.float32)
            
        self.new_data = True
        self.test = test
        self.augment = augment

    def _to3channel(self, x):
        return x.expand(3, *x.shape[0:]).permute(dims=(1, 0, 2, 3))

    def __len__(self):
        if self.new_data == True: 
            self.create_batches(self.batch_size)
        self.new_data = False
        return self.random_idx.shape[0]

    def __iter__(self):        
        self.batch_idx = 0
        return self

    def __next__(self):
        if self.batch_idx < self.random_idx.shape[0]:
            img_o, img_n, era5 = self.get_batch(self.random_idx[self.batch_idx])
            self.batch_idx += 1
            return img_o, img_n, era5
        else:
            raise StopIteration

    def add_rotations(self, n=3):  
        x, y, z = self.img_o, self.img_n, self.era5
        i = 0
        while i < n:
            x_90, y_90, z_90 = rotate90(x, y, z)
            self.img_o, self.img_n, self.era5 = torch.cat([self.img_o, x_90]), torch.cat([self.img_n, y_90]), torch.cat([self.era5, z_90])
            x, y, z = x_90, y_90, z_90
            i += 1

    def add_image(self, img_o, img_n, era5):
        self.img_o = torch.cat((self.img_o, img_o), 0)
        self.img_n = torch.cat((self.img_n, img_n), 0)
        self.era5 = torch.cat((self.era5, era5), 0)
        self.new_data = True

    def normalize(self, img):
        return (img - img.min()) / (img.max() - img.min())
    
    def add_dataloader(self, cyclone_dataloader):
        if self.mode == "sr":
            img_o = self.normalize(cyclone_dataloader.img_64)
            img_n = self.normalize(cyclone_dataloader.img_128)
            era5  = cyclone_dataloader.era5       
        if self.mode == "tp":
            img_o = cyclone_dataloader.img_64
            img_n = cyclone_dataloader.img_128
            era5  = self.normalize(cyclone_dataloader.era5)
        if self.mode == "fc":
            img_o = self.normalize(cyclone_dataloader.img_64)
            img_n = cyclone_dataloader.img_128
            era5  = cyclone_dataloader.era5
        self.add_image(img_o, img_n, era5)

    def create_batches(self, batch_size, perform_augmentation=True):
        if self.augment and perform_augmentation and (self.test == False): self.add_rotations()
        size = self.era5.shape[0]
        if size % batch_size != 0:
            size -= size % batch_size
        idx = torch.tensor(list(range(size)))
        if self.shuffle: idx = torch.randperm(size)
        self.random_idx = idx.reshape(-1, batch_size)

    def get_batch(self, idx): 
        if self.mode == "sr":
            return self._to3channel(self.img_o[idx]), self._to3channel(self.img_n[idx]), self.era5[idx]
        if self.mode == "tp":
            return self.img_o[idx], self.img_n[idx], self._to3channel(self.era5[idx])
        if self.mode == "fc":
            return self._to3channel(self.img_o[idx]), self.img_n[idx], self.era5[idx]

# ---------------------------------------------
# Other helpful methods
# ---------------------------------------------

def get_map_img(m, ax, data, map_bounds, era5=False,
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

    if era5: im = m.imshow(data, origin='upper', extent=map_bounds, cmap='jet', ax=ax)
    elif contour: im = m.contourf(x, y, data, cmap='jet', ax=ax)
    else: im = m.imshow(data, origin='upper', extent=map_bounds, cmap="gray", ax=ax)

    m.drawparallels(range(int(map_bounds[1])-1, int(map_bounds[3]), 7), labels=y_labels, fontsize=10, ax=ax)
    m.drawmeridians(range(int(map_bounds[0])-1, int(map_bounds[2]), 7), labels=x_labels, fontsize=10, ax=ax)

    if title: ax.set_title(title)

    if colorbar:
        fig.colorbar(im, cax=cax, orientation='vertical')    

    return im

def img2req(img):
    mask = img < 1 ; img[mask] = 0
    min_x = torch.min(torch.nonzero(mask)[:, 0])
    min_y = torch.min(torch.nonzero(mask)[:, 1])
    pad = 2
    
    if min_x > min_y:  img = img[0: min_x-pad, :]
    else:  img = img[:, 0: min_y-pad]

    return img

def normalize(img, max_val=None, min_val=None):
    if not max_val or not min_val:
        max_val = img.max()
        min_val = img.min()
    return (img - min_val) / (max_val - min_val)

def unnormalize(img, max_val, min_val):
    return (img * (max_val - min_val)) + min_val

def to3channel(x):
    return x.expand(3, *x.shape[0:]).permute(dims=(1, 0, 2, 3))

def rotate90(x, y, z):
    return torch.rot90(x, dims=[-2, -1]), torch.rot90(y, dims=[-2, -1]), torch.rot90(z, dims=[-2, -1])

def get_bbox_square(x_0, y_0, hs_length):   
    wgs84 = Proj(init='epsg:4326')
    center_x, center_y = wgs84(x_0, y_0)   
    bbox_wgs84 = [
        (center_x - hs_length, center_y - hs_length),  # Bottom left corner
        (center_x + hs_length, center_y - hs_length),  # Bottom right corner
        (center_x + hs_length, center_y + hs_length),  # Top right corner
        (center_x - hs_length, center_y + hs_length),  # Top left corner
        (center_x - hs_length, center_y - hs_length)   # Repeat the first point to close the square
    ]
    
    bbox_utm = []
    for x, y in bbox_wgs84:
        lon, lat = wgs84(x, y, inverse=True)
        bbox_utm.append((lon, lat))
    
    west_lon, south_lat, east_lon, north_lat = bbox_utm[0][0], bbox_utm[0][1], bbox_utm[2][0], bbox_utm[2][1]
    wbox = (west_lon, south_lat, east_lon, north_lat)
    
    return wbox


def np64_to_datetime(date):
    """
    Converts a numpy datetime64 object to a python datetime object 
    Input:
      date - a np.datetime64 object
    Output:
      DATE - a python datetime object
    """
    timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    return datetime.utcfromtimestamp(timestamp)

def get_cropped_era5_ds(ds, wbox_bounds):
    min_lon = wbox_bounds[0]
    min_lat = wbox_bounds[1]
    max_lon = wbox_bounds[2]
    max_lat = wbox_bounds[3]

    mask_lon = (ds.longitude >= min_lon) & (ds.longitude <= max_lon)
    mask_lat = (ds.latitude >= min_lat) & (ds.latitude <= max_lat)
    
    return ds.where(mask_lon & mask_lat, drop=True)

def get_insat3d_ir108_scn(h5_file, map_bounds):
    scn = satpy.Scene(reader="insat3d_img_l1b_h5", 
                      filenames=[h5_file])
    scn.load(['TIR1'])
    mc_scn = scn.crop(ll_bbox=map_bounds).resample(resampler='native')
    ir108_scn = mc_scn['TIR1']
    return ir108_scn

def get_himawari_ir108_scn(hr_dir, map_bounds):
    bz2_files = get_bz2_files(hr_dir)
    filenames = [satpy.readers.FSFile(fsspec.open(x, compression="bz2")) for x in bz2_files]
    scn = satpy.Scene(reader="ahi_hsd", filenames=filenames)
    scn.load(['B13'])
    mc_scn = scn.crop(ll_bbox=map_bounds).resample(resampler='native')
    ir108_scn = mc_scn['B13']
    return ir108_scn
    
def get_era5_map(era5_nc_files, map_bounds=None):
    era5 = xarray.open_mfdataset(era5_nc_files)
    if map_bounds:
        mc_era5 = get_cropped_era5_ds(era5, map_bounds)
        lat = mc_era5.variables["latitude"][:]
        lon = mc_era5.variables["longitude"][:]
        era5_map_bounds = [np.min(lon).values, 
                      np.min(lat).values, 
                      np.max(lon).values, 
                      np.max(lat).values]
        return mc_era5, era5_map_bounds
    return era5

def rev_scn(scn):
    return scn.isel(y=slice(None, None, -1)).isel(x=slice(None, None, -1))

def get_msg_ir108_scn(nat_file, map_bounds):
    scn = satpy.Scene(reader="seviri_l1b_native", 
                  filenames=[nat_file])
    scn.load(['IR_108'])
    mc_scn = scn.crop(ll_bbox=map_bounds).resample(resampler='native')
    ir108_scn = rev_scn(mc_scn['IR_108'])
    return ir108_scn

def get_goes_ir108_scn(filename, map_bounds):
    scn = satpy.Scene(reader="abi_l1b", 
                  filenames=[filename])
    scn.load(['C13'])
    mc_scn = scn.crop(ll_bbox=map_bounds).resample(resampler='native')
    ir108_scn = mc_scn['C13']
    return ir108_scn

def round_to_closest_hour(t):
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
               +timedelta(hours=t.minute//30))

def get_bz2_files(folder):
    return sorted(glob.glob(f"{folder}/*.bz2"))

def transform_make_sq_image(img):
    max_px = max(img.shape)
    img = np.pad(img, ((0, max_px - img.shape[0]), (0, max_px - img.shape[1])), 
             mode='constant')
    return img