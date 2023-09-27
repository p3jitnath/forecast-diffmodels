######################
# Code for Diffusion Models
# Ref: https://github.com/dome272/Diffusion-Models-pytorch
######################

import os
import glob
import sys
import pickle
import torch
import torchvision
import torch.nn.functional as F
import torchmetrics
import torchvision.transforms as T
from pixelmatch.contrib.PIL import pixelmatch

from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.append("/homes/pn222/Work/MSc_Project_pn222/dataproc")
from utils import *

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def pixeldiff(y_true, y_pred):
    transform = T.ToPILImage()
    img_a = transform(y_true)
    img_b = transform(y_pred) 
    img_diff = Image.new("RGBA", img_a.size)
    _ = pixelmatch(img_a, img_b, img_diff, alpha=1)
    return img_diff

def save_images_v2(dataloader, x, y, path=None, **kwargs):
    x = dataloader.normalize(x).cpu()
    y = dataloader.normalize(y).cpu()
    ncols = min(4, len(x))
    
    fig, axes = plt.subplots(3, ncols, figsize=(2*ncols, 6))
    for i in range(0, ncols):
        axes[0, i].imshow(x[i].permute(1, 2, 0)[:, :, 0], cmap="gray")  
        axes[1, i].imshow(y[i].permute(1, 2, 0)[:, :, 0], cmap="gray")
        axes[2, i].imshow(pixeldiff(x[i], y[i]))
    for row in axes:
        for ax in row: 
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
    fig.subplots_adjust(wspace=0, hspace=0)
    if path:
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    
def get_cifar10_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    if args.experiment and args.subset_data:
        dataset = torch.utils.data.Subset(dataset, indices=range(args.subset_data))
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True,
                            num_workers=8)
    return dataloader

def get_satellite_data(args):
    if hasattr(args, "region"):
        c_dataloader_fns = glob.glob(args.dataset_path + f"/{args.region}_*.dat") 
    elif hasattr(args, "exclude_region"):
        c_dataloader_fns = glob.glob(f"{args.dataset_path}/[!{args.exclude_region}]*.dat")
    else:
        c_dataloader_fns = glob.glob(args.dataset_path + f"/*.dat")

    test_set = pickle.load(open("/homes/pn222/Work/MSc_Project_pn222/dataproc/test_set.pkl", "rb"))

    if hasattr(args, "augment"): augment = args.augment
    else: augment = False
    
    if hasattr(args, "mode"): mode = args.mode
    else: mode = "sr"
    
    train_dataloader = ModelDataLoader(batch_size=args.batch_size, 
                                       o_size=args.o_size, 
                                       n_size=args.n_size,
                                       augment=augment,
                                       mode=mode)
    test_dataloader  = ModelDataLoader(batch_size=args.batch_size, 
                                       o_size=args.o_size, 
                                       n_size=args.n_size, 
                                       mode=mode,
                                       test=True)

    if args.datalimit == True:
        c_dataloader_fns = c_dataloader_fns[:10]
    
    for fn in tqdm(c_dataloader_fns):
        with open(fn, "rb") as file: 
            region, name = fn.split('/')[-1][:-4].split('_')
            if name in test_set[region]:
                test_dataloader.add_dataloader(pickle.load(file))
            else:
                train_dataloader.add_dataloader(pickle.load(file))
    
    return train_dataloader, test_dataloader

def setup_logging(run_name, BASE_DIR="."):
    os.makedirs(f"{BASE_DIR}/models", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/results", exist_ok=True)
    os.makedirs(os.path.join(f"{BASE_DIR}/models", run_name), exist_ok=True)
    os.makedirs(os.path.join(f"{BASE_DIR}/results", run_name), exist_ok=True)

def KL_DivLoss(y_pred, y_true):
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    log_input = F.log_softmax(y_pred, dim=1)
    log_target = F.log_softmax(y_true, dim=1)
    output = kl_loss(log_input, log_target)
    return output

def RMSELoss(y_pred, y_true):
    mse_loss = torch.nn.MSELoss(reduction="mean")
    output = torch.sqrt(mse_loss(y_true, y_pred))
    return output

def MAELoss(y_pred, y_true):
    mae_loss = torch.nn.L1Loss(reduction="mean")
    output = torch.sqrt(mae_loss(y_true, y_pred))
    return output  

def PSNR(y_pred, y_true):
    psnr = torchmetrics.PeakSignalNoiseRatio()
    output = psnr(y_pred, y_true)
    return output   

def SSIM(y_pred, y_true):
    ssim = torchmetrics.StructuralSimilarityIndexMeasure()
    output = ssim(y_pred, y_true)
    return output

def FID(y_pred, y_true):
    from torchmetrics.image.fid import FrechetInceptionDistance
    
    fid = FrechetInceptionDistance(feature=64, normalize=True)
    fid.update(y_true, real=True)
    fid.update(y_pred, real=False)
    output = fid.compute()
    return output

def calculate_metrics(y_pred, y_true):
    fn_list = [
        ("kl_div", KL_DivLoss), 
        ("rmse", RMSELoss), 
        ("mae", MAELoss),
        ("psnr", PSNR),
        ("ssim", SSIM),
        ("fid", FID)
    ]
    metric_dict = {}
    for fn_name, fn in fn_list:
        metric_dict[fn_name] = fn(y_pred, y_true)
    return metric_dict
