import argparse
import os
from utils.utils import str2bool, get_models, download_models
from datetime import datetime
import torch
from dataclasses import dataclass
from functools import partial
import cv2
import pandas as pd
import timm
import lpips
from PIL import Image, ImageOps
import requests
from glob import glob
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from datetime import datetime
import matplotlib.pyplot as plt
from functools import partial
from numpy import asarray
from einops import rearrange, repeat
import torchvision
import time
from omegaconf import OmegaConf
import warnings
import sys
# 
# 

# Add 3rd-party methods
sys.path.append('./AdaBins')
sys.path.append('./disco_difussion')
sys.path.append('./guided_diffusion')
sys.path.append('./MiDaS')
sys.path.append('./pytorch3d-lite') 
sys.path.append('./ResizeRight') 

from ResizeRight.resize_right import resize
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from AdaBins.infer import InferenceHelper
from secondary_diffusion_model import SecondaryDiffusionImageNet, SecondaryDiffusionImageNet2
from clip import clip

MAX_ADABINS_AREA = 500000


def parse_args():
    desc = "Pix2PixHD"

    parser = argparse.ArgumentParser(description=desc)

    # Dataset parameters and input paths
    parser.add_argument('--root_path', type=str, default="pwd", help='Path root where inputs and outputs are located. By default is cwd().')
    parser.add_argument('--init_images_path', type=str, default="init_images", help='Folder name for init images')
    parser.add_argument('--images_out_path', type=str, default="images_out", help='Folder name for images out')
    parser.add_argument('--model_path', type=str, default="models", help='Folder name for models')
    parser.add_argument('--pretrained_path', type=str, default="pretrained", help='Folder name for pretrained')
    
    # Model parameters
    parser.add_argument('--midas_model_type', type=str, default="dpt_large", help='Parameter to set MiDaS depth. Options: midas_v21_small, midas_v21, dpt_large, , dpt_hybrid_nyu')

     # Diffusion and Clip Model Settings
    parser.add_argument('--diffusion_model', type=str, default="512x512_diffusion_uncond_finetune_008100", help='Diffusion Model. Options: 256x256_diffusion_uncond, 512x512_diffusion_uncond_finetune_008100')
    parser.add_argument('--use_secondary_model', type=str2bool, nargs='?', const=True, default=True, help="Use secondary model.")
    parser.add_argument('--diffusion_sampling_mode', type=str, default='ddim', help='Diffusion Model. Options: plms,ddim')
    parser.add_argument('--use_checkpoint', type=str2bool, nargs='?', const=True, default=True, help="Use checkpoint. Options: False and True")
    parser.add_argument('--ViTB32', type=str2bool, nargs='?', const=True, default=True, help="Use ViTB32. Options: False and True")
    parser.add_argument('--ViTB16', type=str2bool, nargs='?', const=True, default=True, help="Use ViTB16. Options: False and True")
    parser.add_argument('--ViTL14', type=str2bool, nargs='?', const=True, default=True, help="Use ViTL14. Options: False and True")
    parser.add_argument('--RN101', type=str2bool, nargs='?', const=True, default=False, help="Use RN101. Options: False and True")
    parser.add_argument('--RN50', type=str2bool, nargs='?', const=True, default=True, help="Use RN50. Options: False and True")
    parser.add_argument('--RN50x4', type=str2bool, nargs='?', const=True, default=False, help="Use RN50x4. Options: False and True")
    parser.add_argument('--RN50x64', type=str2bool, nargs='?', const=True, default=False, help="Use RN50x64. Options: False and True")
    
    #If you're having issues with model downloads, check this to compare SHA's:
    parser.add_argument('--check_model_SHA', type=str2bool, nargs='?', const=True, default=False, help="Use check_model_SHA. Options: False and True")



    # Experiment parameters
    parser.add_argument('--experiment_name', type=str, default="", help='A name for the experiment')
    parser.add_argument('--verbose', type=int, default=0, help='Display training time metrics. Yes: 1, No: 2')
    parser.add_argument('--display_step', type=int, default=100, help='Number of step to display images.')
    parser.add_argument('--write_logs_step', type=int, default=100, help='Number of step to display images.')
    parser.add_argument('--resume_training', type=str2bool, nargs='?', const=True, default=False, help="Continue training allows to resume training. You'll need to add experiment name args to identify the experiment to recover.")


    # Warnings parameters
    parser.add_argument('--warnings', type=str2bool, nargs='?', const=False, default=True, help="Show warnings")


    return parser.parse_args()


def main():
    args = parse_args()

    # warnings
    if args.warnings:
        warnings.filterwarnings("ignore")

    # Resume training and experiment name
    args.experiment_name = args.experiment_name if (args.resume_training) else datetime.now().strftime("%Y_%m_%d_%H_%M") + "_" + args.experiment_name    
    
    # Directories config
    args.root_path = os.path.join(os.getcwd(),args.experiment_name) if args.root_path == 'pwd' else os.path.join(args.root_path,args.experiment_name) 
    args.init_images_path = os.path.join(args.root_path, args.init_images_path)
    args.images_out_path = os.path.join(args.root_path, args.images_out_path)
    args.model_path = os.path.join(args.root_path, args.model_path)
    args.pretrained_path = os.path.join(args.root_path, args.pretrained_path)
    os.makedirs(args.init_images_path, exist_ok=True)
    os.makedirs(args.images_out_path, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.pretrained_path, exist_ok=True)

    # get models
    get_models(args)

    # Import devices
    DEVICE = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
    print('Using device:', DEVICE)
    args.device = DEVICE # At least one of the modules expects this name..

    if args.device=='cuda:0':
        if torch.cuda.get_device_capability(DEVICE) == (8,0): ## A100 fix thanks to Emad
            print('Disabling CUDNN for A100 gpu', file=sys.stderr)
        torch.backends.cudnn.enabled = False

    download_models(args)



    model_config = model_and_diffusion_defaults()
    if args.diffusion_model == '512x512_diffusion_uncond_finetune_008100':
        model_config.update({
            'attention_resolutions': '32, 16, 8',
            'class_cond': False,
            'diffusion_steps': 1000, #No need to edit this, it is taken care of later.
            'rescale_timesteps': True,
            'timestep_respacing': 250, #No need to edit this, it is taken care of later.
            'image_size': 512,
            'learn_sigma': True,
            'noise_schedule': 'linear',
            'num_channels': 256,
            'num_head_channels': 64,
            'num_res_blocks': 2,
            'resblock_updown': True,
            'use_checkpoint': args.use_checkpoint,
            'use_fp16': False,
            'use_scale_shift_norm': True,
        })
    elif args.diffusion_model == '256x256_diffusion_uncond':
        model_config.update({
            'attention_resolutions': '32, 16, 8',
            'class_cond': False,
            'diffusion_steps': 1000, #No need to edit this, it is taken care of later.
            'rescale_timesteps': True,
            'timestep_respacing': 250, #No need to edit this, it is taken care of later.
            'image_size': 256,
            'learn_sigma': True,
            'noise_schedule': 'linear',
            'num_channels': 256,
            'num_head_channels': 64,
            'num_res_blocks': 2,
            'resblock_updown': True,
            'use_checkpoint': args.use_checkpoint,
            'use_fp16': False,
            'use_scale_shift_norm': True,
        })

    args.model_default = model_config['image_size']

    if args.use_secondary_model:
        secondary_model = SecondaryDiffusionImageNet2()
        secondary_model.load_state_dict(torch.load(f'{args.model_path}/secondary_model_imagenet_2.pth', map_location='cpu'))
        secondary_model.eval().requires_grad_(False).to(args.device)

    clip_models = []
    if args.ViTB32 is True: clip_models.append(clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(args.device)) 
    if args.ViTB16 is True: clip_models.append(clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(args.device) ) 
    if args.ViTL14 is True: clip_models.append(clip.load('ViT-L/14', jit=False)[0].eval().requires_grad_(False).to(args.device) ) 
    if args.RN50 is True: clip_models.append(clip.load('RN50', jit=False)[0].eval().requires_grad_(False).to(args.device))
    if args.RN50x4 is True: clip_models.append(clip.load('RN50x4', jit=False)[0].eval().requires_grad_(False).to(args.device)) 
    if args.RN50x16 is True: clip_models.append(clip.load('RN50x16', jit=False)[0].eval().requires_grad_(False).to(args.device)) 
    if args.RN50x64 is True: clip_models.append(clip.load('RN50x64', jit=False)[0].eval().requires_grad_(False).to(args.device)) 
    if args.RN101 is True: clip_models.append(clip.load('RN101', jit=False)[0].eval().requires_grad_(False).to(args.device)) 

    normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    lpips_model = lpips.LPIPS(net='vgg').to(args.device)
        
    print('end')


    # # Multiprocessing
    # args.world_size = args.gpus * args.nodes
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '8888'
    # mp.spawn(train_networks, nprocs=args.gpus, args=(args,))   
    
    # #train_networks(args)




if __name__ == '__main__':
    main()