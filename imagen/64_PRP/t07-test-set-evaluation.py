import glob
import copy
import argparse
from functools import partial, partialmethod

import sys
sys.path.append("../")
sys.path.append("../imagen/")
sys.path.append("../../dataproc/")

from helpers import *
from imagen_pytorch import Unet, Imagen, ImagenTrainer, NullUnet
from send_emails import *

seed_value = 42
torch.manual_seed(seed_value)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)

parser = argparse.ArgumentParser()
parser.add_argument('-run_name', help='Specify the run name (for eg. 64_PRP_rot904_3e-4)')
args = parser.parse_args()

sys.stdout = open(f'TEST_METRICS_LOG_{datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}.log','wt')
print = partial(print, flush=True)
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

RUN_NAME = args.run_name
BASE_DIR = f"/vol/bitbucket/pn222/models/{RUN_NAME}/models/64_PRP/"

print(f"Run name: {RUN_NAME}")

best_epoch_dict = {
    "64_PRP_rot904_3e-4": 260,
    "64_PRP_rot904_sep_3e-4": 210
}

unet1 = Unet(
    dim = 32,
    cond_dim = 1024,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 3,
    layer_attns = (False, True, True, True),
)  

unets = [unet1]

class DDPMArgs:
    def __init__(self):
        pass
    
args = DDPMArgs()
args.batch_size = 16
args.image_size = 64 ; args.o_size = 64 ; args.n_size = 128 ;
args.continuous_embed_dim = 64*64*4
args.dataset_path = f"/vol/bitbucket/pn222/satellite/dataloader/{args.o_size}_PRP"
args.datalimit = False
args.lr = 3e-4
args.mode = "tp"

train_dataloader, test_dataloader = get_satellite_data(args)
_ = len(train_dataloader) ; _ = len(test_dataloader)

print("Dataloaders loaded.")

if '1k' in RUN_NAME:
    timesteps = 1000
else:
    timesteps = 250

imagen = Imagen(
    unets = unets,
    image_sizes = (64),
    timesteps = 250,
    cond_drop_prob = 0.1,
    condition_on_continuous = True,
    continuous_embed_dim = args.continuous_embed_dim,
)

metric_dict = {
    "kl_div": [],
    "rmse": [],
    "mae":  [],
    "psnr": [],
    "ssim": [],
    "fid": []
}

test_metric_dict = copy.deepcopy(metric_dict)
best_epoch = best_epoch_dict[RUN_NAME]
ckpt_trainer_path = f"{BASE_DIR}/ckpt_trainer_1_{best_epoch:03}.pt"
trainer = ImagenTrainer(imagen, lr=args.lr, verbose=False).cuda()
trainer.load(ckpt_trainer_path) 

for idx in range(len(test_dataloader)):
    print(f"Evaluating batch idx {idx} ...")
    
    batch_idx = test_dataloader.random_idx[idx]
    img_64, _, img_tp = test_dataloader.get_batch(batch_idx)
    cond_embeds = img_64.reshape(img_64.shape[0], -1).float().cuda()
    ema_sampled_images = imagen.sample(
                batch_size = img_64.shape[0],          
                cond_scale = 3.,
                continuous_embeds=cond_embeds,
                use_tqdm = False
        )
    
    y_true = img_tp.cpu()
    y_pred = ema_sampled_images.cpu()
    metric_dict = calculate_metrics(y_pred, y_true)
    for key in metric_dict.keys():
        test_metric_dict[key].append(metric_dict[key])

with open(f"/vol/bitbucket/pn222/models/{RUN_NAME}/metrics_test.pkl", "wb") as file:
    pickle.dump(test_metric_dict, file)

print(f'Evaluation completed.')

subject = f"[COMPLETED] Test Evaluation Metrics"
message_txt = f"""Metrics Evaluation Completed for {RUN_NAME}"""
send_txt_email(message_txt, subject)