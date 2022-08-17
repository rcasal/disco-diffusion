import argparse
import os
from utils.utils import str2bool, get_models
from datetime import datetime
import torch
from dataclasses import dataclass
from functools import partial
import cv2
import pandas as pd
import gc
import io
import math
import timm
import lpips
from PIL import Image, ImageOps
import requests
from glob import glob
import json
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
import hashlib
from functools import partial
from numpy import asarray
from einops import rearrange, repeat
import torchvision
import time
from omegaconf import OmegaConf
import warnings
import sys
# from IPython import display
# from ipywidgets import Output
# from IPython.display import Image as ipyimg

# Add 3rd-party methods
sys.path.append('./AdaBins')
sys.path.append('./disco_difussion')
sys.path.append('./guided_diffusion')
sys.path.append('./MiDaS')
sys.path.append('./pytorch3d-lite') 
sys.path.append('./ResizeRight') 

from clip import clip
from ResizeRight.resize_right import resize
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from AdaBins.infer import InferenceHelper
from midas_function import init_midas_depth_model


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



    # Experiment parameters
    parser.add_argument('--experiment_name', type=str, default="", help='A name for the experiment')
    parser.add_argument('--verbose', type=int, default=0, help='Display training time metrics. Yes: 1, No: 2')
    parser.add_argument('--display_step', type=int, default=100, help='Number of step to display images.')
    parser.add_argument('--write_logs_step', type=int, default=100, help='Number of step to display images.')
    parser.add_argument('--resume_training', type=str2bool, nargs='?', const=True, default=False, help="Continue training allows to resume training. You'll need to add experiment name args to identify the experiment to recover.")


    # Warnings parameters
    parser.add_argument('--warnings', type=str2bool, nargs='?', const=False, default=True, help="Show warnings")


    """ 
    
    parser.add_argument('--notes', type=str, default="N/A", help='A description of the experiment')
    """
    return parser.parse_args()


def main():
    args = parse_args()

    # warnings
    if args.warnings:
        warnings.filterwarnings("ignore")

    # Resume training and experiment name
    args.experiment_name = args.experiment_name if (args.resume_training) else datetime.now().strftime("%Y_%m_%d_%H_%M") + "_" + args.experiment_name    
    
    # Function to download models and pretrained
    
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

    midas_model, midas_transform, midas_net_w, midas_net_h, midas_resize_mode, midas_normalization = init_midas_depth_model(args)



        



    # # Multiprocessing
    # args.world_size = args.gpus * args.nodes
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '8888'
    # mp.spawn(train_networks, nprocs=args.gpus, args=(args,))   
    
    # #train_networks(args)




if __name__ == '__main__':
    main()