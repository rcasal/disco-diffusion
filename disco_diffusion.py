import argparse
import os
from utils.utils import str2bool, get_models, download_models, do_run
from datetime import datetime
import torch
from dataclasses import dataclass
from functools import partial
import cv2
import pandas as pd
import timm
import gc
import shutil
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
from tqdm.notebook import tqdm
import torchvision
import time
import numpy as np
from omegaconf import OmegaConf
from midas_function import init_midas_depth_model
import warnings
import sys
import pathlib
import subprocess
import disco_xform_utils as dxf
import math
import random
import io
import pytorch3d_lite.py3d_tools as p3dT

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
from animation_utils import parse_key_frames, split_prompts, get_inbetweens
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
    parser.add_argument('--videoFramesFolder', type=str, default="videoFrames", help='Folder name for videoFrames')

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
    parser.add_argument('--RN50x16', type=str2bool, nargs='?', const=True, default=False, help="Use RN50x16. Options: False and True")
    parser.add_argument('--RN50x64', type=str2bool, nargs='?', const=True, default=False, help="Use RN50x64. Options: False and True")
    
    #If you're having issues with model downloads, check this to compare SHA's:
    parser.add_argument('--check_model_SHA', type=str2bool, nargs='?', const=True, default=False, help="Use check_model_SHA. Options: False and True")

    # Basic Settings
    parser.add_argument('--batch_name', type=str, default="bg", help='Batch_name.')
    parser.add_argument('--steps', type=int, default=250, help='Number of steps. Eg. 25,50,100,150,250,500,1000')
    parser.add_argument('--width', type=int, default=1280, help='Image width. Eg. 1280')
    parser.add_argument('--heigth', type=int, default=768, help='Image width. Eg. 768')
    parser.add_argument('--clip_guidance_scale', type=int, default=5000, help='Clip guidance scale. Eg. 5000')
    parser.add_argument('--tv_scale', type=int, default=0, help='TV scale. Eg. 0')
    parser.add_argument('--range_scale', type=int, default=150, help='Range scale. Eg. 150')
    parser.add_argument('--sat_scale', type=int, default=0, help='Sat scale. Eg. 0')
    parser.add_argument('--cutn_batches', type=int, default=4, help='Cutn batches. Eg. 0')
    parser.add_argument('--skip_augs', type=str2bool, nargs='?', const=True, default=False, help="Skip augs. Options: False and True. By default is False.")
    parser.add_argument('--init_image', type=str, default=None, help='Init Image.')
    parser.add_argument('--init_scale', type=int, default=1000, help='Init Scale. Eg. 1000')
    parser.add_argument('--skip_steps', type=int, default=10, help='Skip steps. Eg. 10. Make sure you set skip_steps to ~50 percent of your steps if you want to use an init image.')

    # Animation Settings
    parser.add_argument('--animation_mode', type=str, default='None', help='Init animation_mode. Options are None, 2D, 3D, Video Input. For animation, you probably want to turn `cutn_batches` to 1 to make it quicker.')
    # Video Input settings
    parser.add_argument('--video_name', type=str, default="training.mp4", help='Name for video. It will be saved on root_path.')
    parser.add_argument('--extract_nth_frame', type=int, default=2, help='Extract nth frame.')
    parser.add_argument('--video_init_seed_continuity', type=str2bool, nargs='?', const=True, default=False, help="Video init seed continuity. Options: False and True. By default is False.")

    #2d Animation Settings
    parser.add_argument('--key_frames', type=str2bool, nargs='?', const=True, default=False, help="Key frames. Options: False and True. By default is False.")
    parser.add_argument('--max_frames', type=int, default=10000, help='Max frames.')
    parser.add_argument('--interp_spline', type=str, default="Linear", help='Interp spline. Options: Linear,Quadratic,Cubic. It is recommended not to change.')
    parser.add_argument('--angle', type=str, default="0:(1)", help='Angle. All rotations are provided in degrees')
    parser.add_argument('--zoom', type=str, default="0: (1), 10: (1.05)", help='Zoom is a multiplier of dimensions, 1 is no zoom.')
    parser.add_argument('--translation_x', type=str, default="0: (0)", help='Translation x.')
    parser.add_argument('--translation_y', type=str, default="0: (0)", help='Translation y.')
    parser.add_argument('--translation_z', type=str, default="0: (10.0)", help='Translation z.')
    parser.add_argument('--rotation_3d_x', type=str, default="0: (0)", help='Rotation 3D x.')
    parser.add_argument('--rotation_3d_y', type=str, default="0: (0)", help='Rotation 3D y.')
    parser.add_argument('--rotation_3d_z', type=str, default="0: (0)", help='Rotation 3D z.')
    parser.add_argument('--midas_depth_model', type=str, default="dpt_large", help='Midas depth model.') # Why is twice?
    parser.add_argument('--midas_weight', type=float, default=0.3, help='Midas weight.') 
    parser.add_argument('--near_plane', type=int, default=200, help='Near Plane.') 
    parser.add_argument('--far_plane', type=int, default=10000, help='Far Plane.')
    parser.add_argument('--fov', type=int, default=40, help='Fov.') 
    parser.add_argument('--padding_mode', type=str, default='border', help='Padding mode.') 
    parser.add_argument('--sampling_mode', type=str, default='bicubic', help='Sampling mode.') 

    # Turbo Mode (3D anim only)
    parser.add_argument('--turbo_mode', type=str2bool, nargs='?', const=True, default=False, help="Turbo mode. Options: False and True. By default is False. (Starts after frame 10,) skips diffusion steps and just uses depth map to warp images for skipped frames. Speeds up rendering by 2x-4x, and may improve image coherence between frames. frame_blend_mode smooths abrupt texture changes across 2 frames. For different settings tuned for Turbo Mode, refer to the original Disco-Turbo Github: https://github.com/zippy731/disco-diffusion-turbo")
    parser.add_argument('--turbo_steps', type=str, default='3', help='Turbo steps. Options: 2,3,4,5,6.') 
    parser.add_argument('--turbo_preroll', type=int, default=10, help='Turbo preroll.') 

    # Coherency Settings 
    parser.add_argument('--frames_scale', type=int, default=1500, help='Frames scale. Frame_scale tries to guide the new frame to looking like the old one. A good default is 1500.') 
    parser.add_argument('--frames_skip_steps', type=str, default='60%', help="Frame skip steps` will blur the previous frame - higher values will flicker less but struggle to add enough new detail to zoom into. Options: '40%', '50%', '60%', '70%', '80%'") 

    # VR Mode
    parser.add_argument('--vr_mode', type=str2bool, nargs='?', const=True, default=False, help="VR mode. Options: False and True. By default is False. Enables stereo rendering of left/right eye views (supporting Turbo) which use a different (fish-eye) camera projection matrix. Note the images you're prompting will work better if they have some inherent wide-angle aspect The generated images will need to be combined into left/right videos. These can then be stitched into the VR180 format. Google made the VR180 Creator tool but subsequently stopped supporting it. It's available for download in a few places including https://www.patrickgrunwald.de/vr180-creator-download The tool is not only good for stitching (videos and photos) but also for adding the correct metadata into existing videos, which is needed for services like YouTube to identify the format correctly. Watching YouTube VR videos isn't necessarily the easiest depending on your headset. For instance Oculus have a dedicated media studio and store which makes the files easier to access on a Quest https://creator.oculus.com/manage/mediastudio/. The command to get ffmpeg to concat your frames for each eye is in the form: ffmpeg -framerate 15 -i frame_%4d_l.png l.mp4 (repeat for r).")
    parser.add_argument('--vr_eye_angle', type=float, default=0.5, help='Vr eye angle` is the y-axis rotation of the eyes towards the center.') 
    parser.add_argument('--vr_ipd', type=float, default=5.0, help='Interpupillary distance (between the eyes).') 

    # Extra Settings
    # Saving
    parser.add_argument('--intermediate_saves', type=int, default=100, help='Intermediate saves. Intermediate steps will save a copy at your specified intervals. You can either format it as a single integer or a list of specific steps. A value of 2 will save a copy at 33% and 66%. 0 will save none. A value of [5, 9, 34, 45] will save at steps 5, 9, 34, and 45. (Make sure to include the brackets') 
    parser.add_argument('--intermediates_in_subfolder', type=str2bool, nargs='?', const=True, default=True, help="Intermediates in subfolder. Options: False and True. By default is True.")

    # Advanced Settings
    parser.add_argument('--perlin_init', type=str2bool, nargs='?', const=True, default=False, help="Perlin Init. Options: False and True. By default is False. Perlin init will replace your init, so uncheck if using one.")
    parser.add_argument('--perlin_mode', type=str, default='mixed', help="Perlin Mode. Options: 'mixed', 'color', 'gray'.") 
    parser.add_argument('--set_seed', type=str, default='random_seed', help="Set seed.") 
    parser.add_argument('--eta', type=float, default=0.8, help='Eta.') 
    parser.add_argument('--clamp_grad', type=str2bool, nargs='?', const=True, default=True, help="Clamp grad.")
    parser.add_argument('--clamp_max', type=float, default=0.05, help='Camp max.') 

    # Extra advanced Settings
    parser.add_argument('--randomize_class', type=str2bool, nargs='?', const=True, default=True, help="Randomize class.")
    parser.add_argument('--clip_denoised', type=str2bool, nargs='?', const=True, default=False, help="Clip denoised.")
    parser.add_argument('--fuzzy_prompt', type=str2bool, nargs='?', const=False, default=True, help="Fuzzy prompt.")
    parser.add_argument('--rand_mag', type=float, default=0.05, help='Camp max.') 

    # Cutn Scheduling
    parser.add_argument('--cut_overview', type=str, default="[12]*400+[4]*600", help="cut_overview and cut_innercut are cumulative for total cutn on any given step. Overview cuts see the entire image and are good for early structure, innercuts are your standard cutn.") 
    parser.add_argument('--cut_innercut', type=str, default="[4]*400+[12]*600", help="cut_overview and cut_innercut are cumulative for total cutn on any given step. Overview cuts see the entire image and are good for early structure, innercuts are your standard cutn.") 
    parser.add_argument('--cut_ic_pow', type=float, default=1, help='Cut_ic_pow.') 
    parser.add_argument('--cut_icgray_p', type=str, default="[0.2]*400+[0]*600", help="cut_icgray_p") 

    # Prompts
    parser.add_argument('--text_prompts', type=str, default="A picture of a small tokyo alley way in the style of black and white manga, gray scale.", help="Text prompt") 
    parser.add_argument('--text_prompts_100_0', type=str, default="This set of prompts start at frame 100", help="Text prompt_100_0") 
    parser.add_argument('--text_prompts_100_1', type=str, default="This prompt has weight five:5", help="Text prompt_100_1") 

    parser.add_argument('--image_prompts', type=str, default="None", help="Image Prompts") 

    # Do the run!
    parser.add_argument('--display_rate', type=int, default=50, help='Display Rate') 
    parser.add_argument('--n_batches', type=int, default=1, help='n_batches. It is ignored with animation modes.') 
    parser.add_argument('--resume_run', type=str2bool, nargs='?', const=True, default=False, help="Resume run.")
    parser.add_argument('--run_to_resume', type=str, default="latest", help="Run to resume.") 
    parser.add_argument('--resume_from_frame', type=str, default="latest", help="Resume from frame.") 
    parser.add_argument('--retain_overwritten_frames', type=str2bool, nargs='?', const=True, default=True, help="Retain overwritten frames.")




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
    args.batchFolder = os.path.join(args.images_out_path, args.batch_name)
    os.makedirs(args.batchFolder, exist_ok=True)

    args.video_init_path = os.path.join(args.root_path, args.video_name)

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

    # Download models
    download_models(args)

    # Model config
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

        
    #Get corrected sizes
    args.width_height = [args.width, args.heigth]
    args.side_x = (args.width//64)*64;
    args.side_y = (args.heigth//64)*64;
    if args.side_x != args.width or args.side_y != args.heigth:
        print(f'Changing output size to {args.side_x}x{args.side_y}. Dimensions must by multiples of 64.')

    #Update Model Settings
    args.timestep_respacing = f'ddim{args.steps}'
    args.diffusion_steps = (1000//args.steps)*args.steps if args.steps < 1000 else args.steps
    model_config.update({
        'timestep_respacing': args.timestep_respacing,
        'diffusion_steps': args.diffusion_steps,
        })
    
    # Animation Mode
    if args.animation_mode == "Video Input":
        os.makedirs(os.path.join(args.root_path, args.videoFramesFolder), exist_ok=True)
        print(f"Exporting Video Frames (1 every {args.extract_nth_frame})...")
        try:
            for f in pathlib.Path(f'{args.videoFramesFolder}').glob('*.jpg'):
                f.unlink()
        except:
            print('')
        vf = f'select=not(mod(n\,{args.extract_nth_frame}))'
        subprocess.run(['ffmpeg', '-i', f'{args.video_init_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{args.videoFramesFolder}/%04d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')
        #!ffmpeg -i {video_init_path} -vf {vf} -vsync vfr -q:v 2 -loglevel error -stats {videoFramesFolder}/%04d.jpg

    # Video input 
    if args.animation_mode == "Video Input":
        args.max_frames = len(glob(f'{args.videoFramesFolder}/*.jpg'))

    # Turbo mode. Insist turbo be used only w 3d anim.
    if args.turbo_mode and args.animation_mode != '3D':
        print('=====')
        print('Turbo mode only available with 3D animations. Disabling Turbo.')
        print('=====')
        args.turbo_mode = False

    # VR Mode. Insist VR be used only w 3d anim.
    if args.vr_mode and args.animation_mode != '3D':
        print('=====')
        print('VR mode only available with 3D animations. Disabling VR.')
        print('=====')
        args.vr_mode = False

    # Animation code
    if args.key_frames:
        try:
            args.angle_series = get_inbetweens(parse_key_frames(args.angle), args)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `angle` correctly for key frames.\n"
                "Attempting to interpret `angle` as "
                f'"0: ({args.angle})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            args.angle = f"0: ({args.angle})"
            args.angle_series = get_inbetweens(parse_key_frames(args.angle))

        try:
            args.zoom_series = get_inbetweens(parse_key_frames(args.zoom), args)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `zoom` correctly for key frames.\n"
                "Attempting to interpret `zoom` as "
                f'"0: ({args.zoom})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            args.zoom = f"0: ({args.zoom})"
            args.zoom_series = get_inbetweens(parse_key_frames(args.zoom), args)

        try:
            args.translation_x_series = get_inbetweens(parse_key_frames(args.translation_x), args)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `translation_x` correctly for key frames.\n"
                "Attempting to interpret `translation_x` as "
                f'"0: ({args.translation_x})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            args.translation_x = f"0: ({args.translation_x})"
            args.translation_x_series = get_inbetweens(parse_key_frames(args.translation_x), args)

        try:
            args.translation_y_series = get_inbetweens(parse_key_frames(args.translation_y), args)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `translation_y` correctly for key frames.\n"
                "Attempting to interpret `translation_y` as "
                f'"0: ({args.translation_y})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            args.translation_y = f"0: ({args.translation_y})"
            args.translation_y_series = get_inbetweens(parse_key_frames(args.translation_y), args)

        try:
            args.translation_z_series = get_inbetweens(parse_key_frames(args.translation_z), args)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `translation_z` correctly for key frames.\n"
                "Attempting to interpret `translation_z` as "
                f'"0: ({args.translation_z})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            args.translation_z = f"0: ({args.translation_z})"
            args.translation_z_series = get_inbetweens(parse_key_frames(args.translation_z), args)

        try:
            args.rotation_3d_x_series = get_inbetweens(parse_key_frames(args.rotation_3d_x), args)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `rotation_3d_x` correctly for key frames.\n"
                "Attempting to interpret `rotation_3d_x` as "
                f'"0: ({args.rotation_3d_x})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            args.rotation_3d_x = f"0: ({args.rotation_3d_x})"
            args.rotation_3d_x_series = get_inbetweens(parse_key_frames(args.rotation_3d_x), args)

        try:
            args.rotation_3d_y_series = get_inbetweens(parse_key_frames(args.rotation_3d_y), args)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `rotation_3d_y` correctly for key frames.\n"
                "Attempting to interpret `rotation_3d_y` as "
                f'"0: ({args.rotation_3d_y})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            args.rotation_3d_y = f"0: ({args.rotation_3d_y})"
            args.rotation_3d_y_series = get_inbetweens(parse_key_frames(args.rotation_3d_y), args)

        try:
            args.rotation_3d_z_series = get_inbetweens(parse_key_frames(args.rotation_3d_z), args)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `rotation_3d_z` correctly for key frames.\n"
                "Attempting to interpret `rotation_3d_z` as "
                f'"0: ({args.rotation_3d_z})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            args.rotation_3d_z = f"0: ({args.rotation_3d_z})"
            args.rotation_3d_z_series = get_inbetweens(parse_key_frames(args.rotation_3d_z), args)

    else:
        args.angle = 0
        args.zoom = 0
        args.translation_x = 0
        args.translation_y = 0
        args.translation_z = 0
        args.rotation_3d_x = 0
        args.rotation_3d_y = 0
        args.rotation_3d_z = 0

    # Saving
    if type(args.intermediate_saves) is not list:
        if args.intermediate_saves:
            steps_per_checkpoint = math.floor((args.steps - args.skip_steps - 1) // (args.intermediate_saves+1))
            steps_per_checkpoint = steps_per_checkpoint if steps_per_checkpoint > 0 else 1
            print(f'Will save every {steps_per_checkpoint} steps')
        else:
            steps_per_checkpoint = args.steps+10
    else:
        steps_per_checkpoint = None

    if args.intermediate_saves and args.intermediates_in_subfolder is True:
        args.partialFolder = f'{args.batchFolder}/partials'
        os.makedirs(args.partialFolder, exist_ok=True)

    # Prompt
    args.text_prompts = {
        0: [args.text_prompts],
        100: [args.text_prompts_100_0, args.text_prompts_100_1],
    }

    if args.image_prompts is "None":
        args.image_prompts = {}
    else:
        args.image_prompts = {
            args.image_prompts,
        }


    # Do the run!!
    #Update Model Settings
    args.timestep_respacing = f'ddim{args.steps}'
    args.diffusion_steps = (1000//args.steps)*args.steps if args.steps < 1000 else args.steps
    model_config.update({
        'timestep_respacing': args.timestep_respacing,
        'diffusion_steps': args.diffusion_steps,
    })
    args.batch_size = 1 

    def move_files(start_num, end_num, old_folder, new_folder):
        for i in range(start_num, end_num):
            old_file = old_folder + f'/{args.batch_name}({args.batchNum})_{i:04}.png'
            new_file = new_folder + f'/{args.batch_name}({args.batchNum})_{i:04}.png'
            os.rename(old_file, new_file)

    if args.retain_overwritten_frames is True:
        args.retainFolder = f'{args.batchFolder}/retained'
        os.makedirs(args.retainFolder, exist_ok=True)


    args.skip_step_ratio = int(args.frames_skip_steps.rstrip("%")) / 100
    args.calc_frames_skip_steps = math.floor(args.steps * args.skip_step_ratio)

    if args.steps <= args.calc_frames_skip_steps:
        sys.exit("ERROR: You can't skip more steps than your total steps")

    if args.resume_run:
        if args.run_to_resume == 'latest':
            try:
                args.batchNum
            except:
                args.batchNum = len(glob(f"{args.batchFolder}/{args.batch_name}(*)_settings.txt"))-1
        else:
            args.batchNum = int(args.run_to_resume)
        if args.resume_from_frame == 'latest':
            args.start_frame = len(glob(args.batchFolder+f"/{args.batch_name}({args.batchNum})_*.png"))
            if args.animation_mode != '3D' and args.turbo_mode == True and args.start_frame > args.turbo_preroll and args.start_frame % int(args.turbo_steps) != 0:
                args.start_frame = args.start_frame - (args.start_frame % int(args.turbo_steps))
        else:
            args.start_frame = int(args.resume_from_frame)+1
            if args.animation_mode != '3D' and args.turbo_mode == True and args.start_frame > args.turbo_preroll and args.start_frame % int(args.turbo_steps) != 0:
                args.start_frame = args.start_frame - (args.start_frame % int(args.turbo_steps))
            if args.retain_overwritten_frames is True:
                args.existing_frames = len(glob(args.batchFolder+f"/{args.batch_name}({args.batchNum})_*.png"))
            args.frames_to_save = args.existing_frames - args.start_frame
            print(f'Moving {args.frames_to_save} frames to the Retained folder')
            move_files(args.start_frame, args.existing_frames, args.batchFolder, args.retainFolder)
    else:
        args.start_frame = 0
        args.batchNum = len(glob(args.batchFolder+"/*.txt"))
        while os.path.isfile(f"{args.batchFolder}/{args.batch_name}({args.batchNum})_settings.txt") is True or os.path.isfile(f"{args.batchFolder}/{args.batch_name}-{args.batchNum}_settings.txt") is True:
            args.batchNum += 1

    print(f'Starting Run: {args.batch_name}({args.batchNum}) at frame {args.start_frame}')


    if args.set_seed == 'random_seed':
        random.seed()
        args.seed = random.randint(0, 2**32)
        # print(f'Using seed: {seed}')
    else:
        args.seed = int(args.set_seed)

    args_m = {
        'batchNum': args.batchNum,
        'prompts_series':split_prompts(args.text_prompts, args) if args.text_prompts else None,
        'image_prompts_series':split_prompts(args.image_prompts, args) if args.image_prompts else None,
        'seed': args.seed,
        'display_rate':args.display_rate,
        'n_batches':args.n_batches if args.animation_mode == 'None' else 1,
        'batch_size':args.batch_size,
        'batch_name': args.batch_name,
        'steps': args.steps,
        'diffusion_sampling_mode': args.diffusion_sampling_mode,
        'width_height': args.width_height,
        'clip_guidance_scale': args.clip_guidance_scale,
        'tv_scale': args.tv_scale,
        'range_scale': args.range_scale,
        'sat_scale': args.sat_scale,
        'cutn_batches': args.cutn_batches,
        'init_image': args.init_image,
        'init_scale': args.init_scale,
        'skip_steps': args.skip_steps,
        'side_x': args.side_x,
        'side_y': args.side_y,
        'timestep_respacing': args.timestep_respacing,
        'diffusion_steps': args.diffusion_steps,
        'animation_mode': args.animation_mode,
        'video_init_path': args.video_init_path,
        'extract_nth_frame': args.extract_nth_frame,
        'video_init_seed_continuity': args.video_init_seed_continuity,
        'key_frames': args.key_frames,
        'max_frames': args.max_frames if args.animation_mode != "None" else 1,
        'interp_spline': args.interp_spline,
        'start_frame': args.start_frame,
        'angle': args.angle,
        'zoom': args.zoom,
        'translation_x': args.translation_x,
        'translation_y': args.translation_y,
        'translation_z': args.translation_z,
        'rotation_3d_x': args.rotation_3d_x,
        'rotation_3d_y': args.rotation_3d_y,
        'rotation_3d_z': args.rotation_3d_z,
        'midas_depth_model': args.midas_depth_model,
        'midas_weight': args.midas_weight,
        'near_plane': args.near_plane,
        'far_plane': args.far_plane,
        'fov': args.fov,
        'padding_mode': args.padding_mode,
        'sampling_mode': args.sampling_mode,
        'angle_series':None,
        'zoom_series':None,
        'translation_x_series':None,
        'translation_y_series':None,
        'translation_z_series':None,
        'rotation_3d_x_series':None,
        'rotation_3d_y_series':None,
        'rotation_3d_z_series':None,
        'frames_scale': args.frames_scale,
        'calc_frames_skip_steps': args.calc_frames_skip_steps,
        'skip_step_ratio': args.skip_step_ratio,
        'calc_frames_skip_steps': args.calc_frames_skip_steps,
        'text_prompts': args.text_prompts,
        'image_prompts': args.image_prompts,
        'cut_overview': eval(args.cut_overview),
        'cut_innercut': eval(args.cut_innercut),
        'cut_ic_pow': args.cut_ic_pow,
        'cut_icgray_p': eval(args.cut_icgray_p),
        'intermediate_saves': args.intermediate_saves,
        'intermediates_in_subfolder': args.intermediates_in_subfolder,
        'steps_per_checkpoint': steps_per_checkpoint,
        'perlin_init': args.perlin_init,
        'perlin_mode': args.perlin_mode,
        'set_seed': args.set_seed,
        'eta': args.eta,
        'clamp_grad': args.clamp_grad,
        'clamp_max': args.clamp_max,
        'skip_augs': args.skip_augs,
        'randomize_class': args.randomize_class,
        'clip_denoised': args.clip_denoised,
        'fuzzy_prompt': args.fuzzy_prompt,
        'rand_mag': args.rand_mag,
    }

    args = SimpleNamespace(**args_m)

    print('Prepping model...')
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(torch.load(f'{args.model_path}/{args.diffusion_model}.pt', map_location='cpu'))
    model.requires_grad_(False).eval().to(args.device)
    for name, param in model.named_parameters():
        if 'qkv' in name or 'norm' in name or 'proj' in name:
            param.requires_grad_()
    if model_config['use_fp16']:
        model.convert_to_fp16()

    gc.collect()
    torch.cuda.empty_cache()
    try:
        do_run(model, diffusion, secondary_model, args_m, args)
    except KeyboardInterrupt:
        pass
    finally:
        print('Seed used:', args.seed)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()