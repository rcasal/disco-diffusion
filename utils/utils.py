import subprocess
import argparse
import sys
import os
import pytorch3d_lite.py3d_tools as p3dT
import disco_xform_utils as dxf
import torch 
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
import requests
import cv2
import io
import math
import matplotlib.pyplot as plt
from datetime import datetime
from ResizeRight.resize_right import resize
from midas_function import init_midas_depth_model
from tqdm.notebook import tqdm
import numpy as np
import shutil
import random
from clip import clip
import json
import lpips
import hashlib
from secondary_diffusion_model import alpha_sigma_to_t
import gc
from IPython import display
from ipywidgets import Output
# from IPython.display import Image as ipyimg

def gitclone(url):
  res = subprocess.run(['git', 'clone', url], stdout=subprocess.PIPE).stdout.decode('utf-8')
  print(res)


def pipi(modulestr):
  res = subprocess.run(['pip', 'install', modulestr], stdout=subprocess.PIPE).stdout.decode('utf-8')
  print(res)


def pipie(modulestr):
  res = subprocess.run(['git', 'install', '-e', modulestr], stdout=subprocess.PIPE).stdout.decode('utf-8')
  print(res)


def wget(url, outputdir):
  res = subprocess.run(['wget', url, '-P', f'{outputdir}'], stdout=subprocess.PIPE).stdout.decode('utf-8')
  print(res)


def setting_device(args):
  args.device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
  print('Using device:', args.device)

  if args.device=='cuda:0':
      if torch.cuda.get_device_capability(args.device) == (8,0): ## A100 fix thanks to Emad
          print('Disabling CUDNN for A100 gpu', file=sys.stderr)
      torch.backends.cudnn.enabled = False


def create_dirs(args):
  args.root_path = os.path.join(os.getcwd(),args.experiment_name) if args.root_path == 'pwd' else os.path.join(args.root_path,args.experiment_name) 

  # init_images_path
  args.init_images_path = os.path.join(args.root_path, args.init_images_path)
  os.makedirs(args.init_images_path, exist_ok=True)
  # images_out_path
  args.images_out_path = os.path.join(args.root_path, args.images_out_path)
  os.makedirs(args.images_out_path, exist_ok=True)
  # images_out_path/batch_name
  args.batchFolder = os.path.join(args.images_out_path, args.batch_name)
  os.makedirs(args.batchFolder, exist_ok=True)
  # model_path
  args.model_path = os.path.join(args.root_path, args.model_path)
  os.makedirs(args.model_path, exist_ok=True)
  # pretrained_path
  args.pretrained_path = os.path.join(args.root_path, args.pretrained_path)
  os.makedirs(args.pretrained_path, exist_ok=True)

  args.video_init_path = os.path.join(args.root_path, args.video_name)



def download_models(args,fallback=False):
  # MiDaS
  if not os.path.exists(f'{args.model_path}/dpt_large-midas-2f21e586.pt'):
      wget("https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt", args.model_path)
  else: 
      print("'{args.model_path}/dpt_large-midas-2f21e586.pt' already exists.")

  # AdaBins
  if not os.path.exists('pretrained/AdaBins_nyu.pt'):
    os.makedirs('pretrained', exist_ok=True)
    wget("https://cloudflare-ipfs.com/ipfs/Qmd2mMnDLWePKmgfS8m6ntAg4nhV5VkUyAydYBp8cWWeB7/AdaBins_nyu.pt", 'pretrained')
  else: 
      print("'pretrained/AdaBins_nyu.pt' already exists.")


  model_256_downloaded = False
  model_512_downloaded = False
  model_secondary_downloaded = False

  model_256_SHA = '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a'
  model_512_SHA = '9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648'
  model_secondary_SHA = '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a'

  model_256_link = 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'
  model_512_link = 'https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt'
  model_secondary_link = 'https://huggingface.co/spaces/huggi/secondary_model_imagenet_2.pth/resolve/main/secondary_model_imagenet_2.pth'

  model_256_link_fb = 'https://www.dropbox.com/s/9tqnqo930mpnpcn/256x256_diffusion_uncond.pt'
  model_512_link_fb = 'https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt'
  model_secondary_link_fb = 'https://ipfs.pollinations.ai/ipfs/bafybeibaawhhk7fhyhvmm7x24zwwkeuocuizbqbcg5nqx64jq42j75rdiy/secondary_model_imagenet_2.pth'

  model_256_path = f'{args.model_path}/256x256_diffusion_uncond.pt'
  model_512_path = f'{args.model_path}/512x512_diffusion_uncond_finetune_008100.pt'
  model_secondary_path = f'{args.model_path}/secondary_model_imagenet_2.pth'

  if fallback:
    model_256_link = model_256_link_fb
    model_512_link = model_512_link_fb
    model_secondary_link = model_secondary_link_fb
  # Download the diffusion model
  if args.diffusion_model == '256x256_diffusion_uncond':
    if os.path.exists(model_256_path) and args.check_model_SHA:
      print('Checking 256 Diffusion File')
      with open(model_256_path,"rb") as f:
          bytes = f.read() 
          hash = hashlib.sha256(bytes).hexdigest();
      if hash == model_256_SHA:
        print('256 Model SHA matches')
        model_256_downloaded = True
      else: 
        print("256 Model SHA doesn't match, redownloading...")
        wget(model_256_link, args.model_path)
        if os.path.exists(model_256_path):
          model_256_downloaded = True
        else:
          print('First URL Failed using FallBack')
          download_models(args.diffusion_model,args.use_secondary_model,True)
    elif os.path.exists(model_256_path) and not args.check_model_SHA or model_256_downloaded == True:
      print('256 Model already downloaded, check check_model_SHA if the file is corrupt')
    else:  
      wget(model_256_link, args.model_path)
      if os.path.exists(model_256_path):
        model_256_downloaded = True
      else:
        print('First URL Failed using FallBack')
        download_models(args.diffusion_model,True)
  elif args.diffusion_model == '512x512_diffusion_uncond_finetune_008100':
    if os.path.exists(model_512_path) and args.check_model_SHA:
      print('Checking 512 Diffusion File')
      with open(model_512_path,"rb") as f:
          bytes = f.read() 
          hash = hashlib.sha256(bytes).hexdigest();
      if hash == model_512_SHA:
        print('512 Model SHA matches')
        if os.path.exists(model_512_path):
          model_512_downloaded = True
        else:
          print('First URL Failed using FallBack')
          download_models(args,True)
      else:  
        print("512 Model SHA doesn't match, redownloading...")
        wget(model_512_link, args.model_path)
        if os.path.exists(model_512_path):
          model_512_downloaded = True
        else:
          print('First URL Failed using FallBack')
          download_models(args,True)
    elif os.path.exists(model_512_path) and not args.check_model_SHA or model_512_downloaded == True:
      print('512 Model already downloaded, check check_model_SHA if the file is corrupt')
    else:  
      wget(model_512_link, args.model_path)
      model_512_downloaded = True
  # Download the secondary diffusion model v2
  if args.use_secondary_model == True:
    if os.path.exists(model_secondary_path) and args.check_model_SHA:
      print('Checking Secondary Diffusion File')
      with open(model_secondary_path,"rb") as f:
          bytes = f.read() 
          hash = hashlib.sha256(bytes).hexdigest();
      if hash == model_secondary_SHA:
        print('Secondary Model SHA matches')
        model_secondary_downloaded = True
      else:  
        print("Secondary Model SHA doesn't match, redownloading...")
        wget(model_secondary_link, args.model_path)
        if os.path.exists(model_secondary_path):
          model_secondary_downloaded = True
        else:
          print('First URL Failed using FallBack')
          download_models(args,True)
    elif os.path.exists(model_secondary_path) and not args.check_model_SHA or model_secondary_downloaded == True:
      print('Secondary Model already downloaded, check check_model_SHA if the file is corrupt')
    else:  
      wget(model_secondary_link, args.model_path)
      if os.path.exists(model_secondary_path):
          model_secondary_downloaded = True
      else:
        print('First URL Failed using FallBack')
        download_models(args,True)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869


# More utils

def interp(t):
  return 3 * t**2 - 2 * t ** 3

def perlin(width, height, scale=10, device=None):
  gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
  xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
  ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
  wx = 1 - interp(xs)
  wy = 1 - interp(ys)
  dots = 0
  dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
  dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
  dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
  dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
  return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)

def perlin_ms(octaves, width, height, grayscale, device=None):
  out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
  # out_array = [0.0] if grayscale else [0.0, 0.0, 0.0]
  for i in range(1 if grayscale else 3):
    scale = 2 ** len(octaves)
    oct_width = width
    oct_height = height
    for oct in octaves:
      p = perlin(oct_width, oct_height, scale, device)
      out_array[i] += p * oct
      scale //= 2
      oct_width *= 2
      oct_height *= 2
  return torch.cat(out_array)

def create_perlin_noise(args, octaves=[1, 1, 1, 1], width=2, height=2, grayscale=True):
  out = perlin_ms(octaves, width, height, grayscale)
  if grayscale:
    out = TF.resize(size=(args.side_y, args.side_x), img=out.unsqueeze(0))
    out = TF.to_pil_image(out.clamp(0, 1)).convert('RGB')
  else:
    out = out.reshape(-1, 3, out.shape[0]//3, out.shape[1])
    out = TF.resize(size=(args.side_y, args.side_x), img=out)
    out = TF.to_pil_image(out.clamp(0, 1).squeeze())

  out = ImageOps.autocontrast(out)
  return out

def regen_perlin(args):
    if args.perlin_mode == 'color':
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, False)
    elif args.perlin_mode == 'gray':
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, True)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)
    else:
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)

    init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(args.device).unsqueeze(0).mul(2).sub(1)
    del init2
    return init.expand(args.batch_size, -1, -1, -1)

def fetch(url_or_path):
  if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
    r = requests.get(url_or_path)
    r.raise_for_status()
    fd = io.BytesIO()
    fd.write(r.content)
    fd.seek(0)
    return fd
  return open(url_or_path, 'rb')

def read_image_workaround(path):
  """OpenCV reads images as BGR, Pillow saves them as RGB. Work around
  this incompatibility to avoid colour inversions."""
  im_tmp = cv2.imread(path)
  return cv2.cvtColor(im_tmp, cv2.COLOR_BGR2RGB)

def parse_prompt(prompt):
  if prompt.startswith('http://') or prompt.startswith('https://'):
    vals = prompt.rsplit(':', 2)
    vals = [vals[0] + ':' + vals[1], *vals[2:]]
  else:
    vals = prompt.rsplit(':', 1)
  vals = vals + ['', '1'][len(vals):]
  return vals[0], float(vals[1])

def sinc(x):
  return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))

def lanczos(x, a):
  cond = torch.logical_and(-a < x, x < a)
  out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
  return out / out.sum()

def ramp(ratio, width):
  n = math.ceil(width / ratio + 1)
  out = torch.empty([n])
  cur = 0
  for i in range(out.shape[0]):
    out[i] = cur
    cur += ratio
  return torch.cat([-out[1:].flip([0]), out])[1:-1]

def resample(input, size, align_corners=True):
  n, c, h, w = input.shape
  dh, dw = size

  input = input.reshape([n * c, 1, h, w])

  if dh < h:
    kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
    pad_h = (kernel_h.shape[0] - 1) // 2
    input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
    input = F.conv2d(input, kernel_h[None, None, :, None])

  if dw < w:
    kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
    pad_w = (kernel_w.shape[0] - 1) // 2
    input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
    input = F.conv2d(input, kernel_w[None, None, None, :])

  input = input.reshape([n, c, h, w])
  return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)

class MakeCutouts(nn.Module):
  def __init__(self, cut_size, cutn, skip_augs=False):
    super().__init__()
    self.cut_size = cut_size
    self.cutn = cutn
    self.skip_augs = skip_augs
    self.augs = T.Compose([
      T.RandomHorizontalFlip(p=0.5),
      T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
      T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
      T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
      T.RandomPerspective(distortion_scale=0.4, p=0.7),
      T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
      T.RandomGrayscale(p=0.15),
      T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
      # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ])

  def forward(self, input):
    input = T.Pad(input.shape[2]//4, fill=0)(input)
    sideY, sideX = input.shape[2:4]
    max_size = min(sideX, sideY)

    cutouts = []
    for ch in range(self.cutn):
      if ch > self.cutn - self.cutn//4:
        cutout = input.clone()
      else:
        size = int(max_size * torch.zeros(1,).normal_(mean=.8, std=.3).clip(float(self.cut_size/max_size), 1.))
        offsetx = torch.randint(0, abs(sideX - size + 1), ())
        offsety = torch.randint(0, abs(sideY - size + 1), ())
        cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]

      if not self.skip_augs:
        cutout = self.augs(cutout)
      cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
      del cutout

    cutouts = torch.cat(cutouts, dim=0)
    return cutouts

cutout_debug = False
padargs = {}


class MakeCutoutsDango(nn.Module):
    def __init__(self, args, cut_size,
                 Overview=4, 
                 InnerCrop = 0, IC_Size_Pow=0.5, IC_Grey_P = 0.2
                 ):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        self.args = args
        if args.animation_mode == 'None':
          self.augs = T.Compose([
              T.RandomHorizontalFlip(p=0.5),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomAffine(degrees=10, translate=(0.05, 0.05),  interpolation = T.InterpolationMode.BILINEAR),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomGrayscale(p=0.1),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
          ])
        elif args.animation_mode == 'Video Input':
          self.augs = T.Compose([
              T.RandomHorizontalFlip(p=0.5),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomPerspective(distortion_scale=0.4, p=0.7),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomGrayscale(p=0.15),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
          ])
        elif  args.animation_mode == '2D' or args.animation_mode == '3D':
          self.augs = T.Compose([
              T.RandomHorizontalFlip(p=0.4),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomAffine(degrees=10, translate=(0.05, 0.05),  interpolation = T.InterpolationMode.BILINEAR),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomGrayscale(p=0.1),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.3),
          ])
          

    def forward(self, input):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        l_size = max(sideX, sideY)
        output_shape = [1,3,self.cut_size,self.cut_size] 
        output_shape_2 = [1,3,self.cut_size+2,self.cut_size+2]
        pad_input = F.pad(input,((sideY-max_size)//2,(sideY-max_size)//2,(sideX-max_size)//2,(sideX-max_size)//2), **padargs)
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview>0:
            if self.Overview<=4:
                if self.Overview>=1:
                    cutouts.append(cutout)
                if self.Overview>=2:
                    cutouts.append(gray(cutout))
                if self.Overview>=3:
                    cutouts.append(TF.hflip(cutout))
                if self.Overview==4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

            if cutout_debug:
                TF.to_pil_image(cutouts[0].clamp(0, 1).squeeze(0)).save("cutout_overview0.jpg",quality=99)

                              
        if self.InnerCrop >0:
            for i in range(self.InnerCrop):
                size = int(torch.rand([])**self.IC_Size_Pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
            if cutout_debug:
                TF.to_pil_image(cutouts[-1].clamp(0, 1).squeeze(0)).save("cutout_InnerCrop.jpg",quality=99)
        cutouts = torch.cat(cutouts)
        if self.args.skip_augs is not True: cutouts=self.augs(cutouts)
        return cutouts

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)     

def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])

stop_on_next_loop = False  # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete
TRANSLATION_SCALE = 1.0/200.0

def do_3d_step(img_filepath, frame_num, midas_model, midas_transform, args):
  if args.key_frames:
    translation_x = args.translation_x_series[frame_num]
    translation_y = args.translation_y_series[frame_num]
    translation_z = args.translation_z_series[frame_num]
    rotation_3d_x = args.rotation_3d_x_series[frame_num]
    rotation_3d_y = args.rotation_3d_y_series[frame_num]
    rotation_3d_z = args.rotation_3d_z_series[frame_num]
    print(
        f'translation_x: {translation_x}',
        f'translation_y: {translation_y}',
        f'translation_z: {translation_z}',
        f'rotation_3d_x: {rotation_3d_x}',
        f'rotation_3d_y: {rotation_3d_y}',
        f'rotation_3d_z: {rotation_3d_z}',
    )

  translate_xyz = [-translation_x*TRANSLATION_SCALE, translation_y*TRANSLATION_SCALE, -translation_z*TRANSLATION_SCALE]
  rotate_xyz_degrees = [rotation_3d_x, rotation_3d_y, rotation_3d_z]
  print('translation:',translate_xyz)
  print('rotation:',rotate_xyz_degrees)
  rotate_xyz = [math.radians(rotate_xyz_degrees[0]), math.radians(rotate_xyz_degrees[1]), math.radians(rotate_xyz_degrees[2])]
  rot_mat = p3dT.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=args.device), "XYZ").unsqueeze(0)
  print("rot_mat: " + str(rot_mat))
  next_step_pil = dxf.transform_image_3d(img_filepath, midas_model, midas_transform, args.device,
                                          rot_mat, translate_xyz, args.near_plane, args.far_plane,
                                          args.fov, padding_mode=args.padding_mode,
                                          sampling_mode=args.sampling_mode, midas_weight=args.midas_weight)
  return next_step_pil

def do_run(model, diffusion, secondary_model, args, args_exp):
  seed = args.seed
  print(range(args.start_frame, args.max_frames))
  normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
  lpips_model = lpips.LPIPS(net='vgg').to(args_exp.device)

  if (args.animation_mode == "3D") and (args.midas_weight > 0.0):
      midas_model, midas_transform, midas_net_w, midas_net_h, midas_resize_mode, midas_normalization = init_midas_depth_model(args.midas_depth_model)
  for frame_num in range(args.start_frame, args.max_frames):
      if stop_on_next_loop:
        break
      
      display.clear_output(wait=True)

      # Print Frame progress if animation mode is on
      if args.animation_mode != "None":
        batchBar = tqdm(range(args.max_frames), desc ="Frames")
        batchBar.n = frame_num
        batchBar.refresh()

      
      # Inits if not video frames
      if args.animation_mode != "Video Input":
        if args.init_image in ['','none', 'None', 'NONE']:
          init_image = None
        else:
          init_image = args.init_image
        init_scale = args.init_scale
        skip_steps = args.skip_steps

      if args.animation_mode == "2D":
        if args.key_frames:
          angle = args.angle_series[frame_num]
          zoom = args.zoom_series[frame_num]
          translation_x = args.translation_x_series[frame_num]
          translation_y = args.translation_y_series[frame_num]
          print(
              f'angle: {angle}',
              f'zoom: {zoom}',
              f'translation_x: {translation_x}',
              f'translation_y: {translation_y}',
          )
        
        if frame_num > 0:
          seed += 1
          if args_exp.resume_run and frame_num == args_exp.start_frame:
            img_0 = cv2.imread(args_exp.batchFolder+f"/{args_exp.batch_name}({args_exp.batchNum})_{args_exp.start_frame-1:04}.png")
          else:
            img_0 = cv2.imread('prevFrame.png')
          center = (1*img_0.shape[1]//2, 1*img_0.shape[0]//2)
          trans_mat = np.float32(
              [[1, 0, translation_x],
              [0, 1, translation_y]]
          )
          rot_mat = cv2.getRotationMatrix2D( center, angle, zoom )
          trans_mat = np.vstack([trans_mat, [0,0,1]])
          rot_mat = np.vstack([rot_mat, [0,0,1]])
          transformation_matrix = np.matmul(rot_mat, trans_mat)
          img_0 = cv2.warpPerspective(
              img_0,
              transformation_matrix,
              (img_0.shape[1], img_0.shape[0]),
              borderMode=cv2.BORDER_WRAP
          )

          cv2.imwrite('prevFrameScaled.png', img_0)
          init_image = 'prevFrameScaled.png'
          init_scale = args.frames_scale
          skip_steps = args.calc_frames_skip_steps

      if args.animation_mode == "3D":
        if frame_num > 0:
          seed += 1    
          if args_exp.resume_run and frame_num == args_exp.start_frame:
            img_filepath = args_exp.batchFolder+f"/{args_exp.batch_name}({args_exp.batchNum})_{args_exp.start_frame-1:04}.png"
            if args_exp.turbo_mode and frame_num > args_exp.turbo_preroll:
              shutil.copyfile(img_filepath, 'oldFrameScaled.png')
          else:
            img_filepath = 'prevFrame.png'

          next_step_pil = do_3d_step(img_filepath, frame_num, midas_model, midas_transform)
          next_step_pil.save('prevFrameScaled.png')

          ### Turbo mode - skip some diffusions, use 3d morph for clarity and to save time
          if args_exp.turbo_mode:
            if frame_num == args_exp.turbo_preroll: #start tracking oldframe
              next_step_pil.save('oldFrameScaled.png')#stash for later blending          
            elif frame_num > args_exp.turbo_preroll:
              #set up 2 warped image sequences, old & new, to blend toward new diff image
              old_frame = do_3d_step('oldFrameScaled.png', frame_num, midas_model, midas_transform)
              old_frame.save('oldFrameScaled.png')
              if frame_num % int(args_exp.turbo_steps) != 0: 
                print('turbo skip this frame: skipping clip diffusion steps')
                filename = f'{args.batch_name}({args.batchNum})_{frame_num:04}.png'
                blend_factor = ((frame_num % int(args_exp.turbo_steps))+1)/int(args_exp.turbo_steps)
                print('turbo skip this frame: skipping clip diffusion steps and saving blended frame')
                newWarpedImg = cv2.imread('prevFrameScaled.png')#this is already updated..
                oldWarpedImg = cv2.imread('oldFrameScaled.png')
                blendedImage = cv2.addWeighted(newWarpedImg, blend_factor, oldWarpedImg,1-blend_factor, 0.0)
                cv2.imwrite(f'{args_exp.batchFolder}/{filename}',blendedImage)
                next_step_pil.save(f'{img_filepath}') # save it also as prev_frame to feed next iteration
                if args_exp.vr_mode:
                  generate_eye_views(TRANSLATION_SCALE,args_exp.batchFolder,filename,frame_num,midas_model, midas_transform)
                continue
              else:
                #if not a skip frame, will run diffusion and need to blend.
                oldWarpedImg = cv2.imread('prevFrameScaled.png')
                cv2.imwrite(f'oldFrameScaled.png',oldWarpedImg)#swap in for blending later 
                print('clip/diff this frame - generate clip diff image')

          init_image = 'prevFrameScaled.png'
          init_scale = args.frames_scale
          skip_steps = args.calc_frames_skip_steps

      if  args.animation_mode == "Video Input":
        if not args_exp.video_init_seed_continuity:
          seed += 1
        init_image = f'{args_exp.videoFramesFolder}/{frame_num+1:04}.jpg'
        init_scale = args.frames_scale
        skip_steps = args.calc_frames_skip_steps

      loss_values = []
  
      if seed is not None:
          np.random.seed(seed)
          random.seed(seed)
          torch.manual_seed(seed)
          torch.cuda.manual_seed_all(seed)
          torch.backends.cudnn.deterministic = True
  
      target_embeds, weights = [], []
      
      if args.prompts_series is not None and frame_num >= len(args.prompts_series):
        frame_prompt = args.prompts_series[-1]
      elif args.prompts_series is not None:
        frame_prompt = args.prompts_series[frame_num]
      else:
        frame_prompt = []
      
      print(args.image_prompts_series)
      if args.image_prompts_series is not None and frame_num >= len(args.image_prompts_series):
        image_prompt = args.image_prompts_series[-1]
      elif args.image_prompts_series is not None:
        image_prompt = args.image_prompts_series[frame_num]
      else:
        image_prompt = []

      print(f'Frame {frame_num} Prompt: {frame_prompt}')

      model_stats = []
      for clip_model in args_exp.clip_models:
            cutn = 16
            model_stat = {"clip_model":None,"target_embeds":[],"make_cutouts":None,"weights":[]}
            model_stat["clip_model"] = clip_model
            
            
            for prompt in frame_prompt:
                txt, weight = parse_prompt(prompt)
                txt = clip_model.encode_text(clip.tokenize(prompt).to(args_exp.device)).float()
                
                if args.fuzzy_prompt:
                    for i in range(25):
                        model_stat["target_embeds"].append((txt + torch.randn(txt.shape).cuda() * args.rand_mag).clamp(0,1))
                        model_stat["weights"].append(weight)
                else:
                    model_stat["target_embeds"].append(txt)
                    model_stat["weights"].append(weight)
        
            if image_prompt:
              model_stat["make_cutouts"] = MakeCutouts(clip_model.visual.input_resolution, cutn, skip_augs=args_exp.skip_augs) 
              for prompt in image_prompt:
                  path, weight = parse_prompt(prompt)
                  img = Image.open(fetch(path)).convert('RGB')
                  img = TF.resize(img, min(args_exp.side_x, args_exp.side_y, *img.size), T.InterpolationMode.LANCZOS)
                  batch = model_stat["make_cutouts"](TF.to_tensor(img).to(args_exp.device).unsqueeze(0).mul(2).sub(1))
                  embed = clip_model.encode_image(normalize(batch)).float()
                  if args_exp.fuzzy_prompt:
                      for i in range(25):
                          model_stat["target_embeds"].append((embed + torch.randn(embed.shape).cuda() * args_exp.rand_mag).clamp(0,1))
                          weights.extend([weight / cutn] * cutn)
                  else:
                      model_stat["target_embeds"].append(embed)
                      model_stat["weights"].extend([weight / cutn] * cutn)
        
            model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
            model_stat["weights"] = torch.tensor(model_stat["weights"], device=args_exp.device)
            if model_stat["weights"].sum().abs() < 1e-3:
                raise RuntimeError('The weights must not sum to 0.')
            model_stat["weights"] /= model_stat["weights"].sum().abs()
            model_stats.append(model_stat)
  
      init = None
      if init_image is not None:
          init = Image.open(fetch(init_image)).convert('RGB')
          init = init.resize((args.side_x, args.side_y), Image.LANCZOS)
          init = TF.to_tensor(init).to(args_exp.device).unsqueeze(0).mul(2).sub(1)
      
      if args.perlin_init:
          if args.perlin_mode == 'color':
              init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
              init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, False)
          elif args.perlin_mode == 'gray':
            init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, True)
            init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)
          else:
            init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
            init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)
          # init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device)
          init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(args_exp.device).unsqueeze(0).mul(2).sub(1)
          del init2
  
      cur_t = None
  
      def cond_fn(x, t, y=None):
          with torch.enable_grad():
              x_is_NaN = False
              x = x.detach().requires_grad_()
              n = x.shape[0]
              if args_exp.use_secondary_model is True:
                alpha = torch.tensor(diffusion.sqrt_alphas_cumprod[cur_t], device=args_exp.device, dtype=torch.float32)
                sigma = torch.tensor(diffusion.sqrt_one_minus_alphas_cumprod[cur_t], device=args_exp.device, dtype=torch.float32)
                cosine_t = alpha_sigma_to_t(alpha, sigma)
                out = secondary_model(x, cosine_t[None].repeat([n])).pred
                fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)
              else:
                my_t = torch.ones([n], device=args_exp.device, dtype=torch.long) * cur_t
                out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
                fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out['pred_xstart'] * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)
              for model_stat in model_stats:
                for i in range(args.cutn_batches):
                    t_int = int(t.item())+1 #errors on last step without +1, need to find source
                    #when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                    try:
                        input_resolution=model_stat["clip_model"].visual.input_resolution
                    except:
                        input_resolution=224

                    cuts = MakeCutoutsDango(args=args, cut_size=input_resolution,
                            Overview= args.cut_overview[1000-t_int], 
                            InnerCrop = args.cut_innercut[1000-t_int], IC_Size_Pow=args.cut_ic_pow, IC_Grey_P = args.cut_icgray_p[1000-t_int]
                            )
                    clip_in = normalize(cuts(x_in.add(1).div(2)))
                    image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                    dists = spherical_dist_loss(image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0))
                    dists = dists.view([args.cut_overview[1000-t_int]+args.cut_innercut[1000-t_int], n, -1])
                    losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                    loss_values.append(losses.sum().item()) # log loss, probably shouldn't do per cutn_batch
                    x_in_grad += torch.autograd.grad(losses.sum() * args_exp.clip_guidance_scale, x_in)[0] / args_exp.cutn_batches
              tv_losses = tv_loss(x_in)
              if args_exp.use_secondary_model is True:
                range_losses = range_loss(out)
              else:
                range_losses = range_loss(out['pred_xstart'])
              sat_losses = torch.abs(x_in - x_in.clamp(min=-1,max=1)).mean()
              loss = tv_losses.sum() * args_exp.tv_scale + range_losses.sum() * args_exp.range_scale + sat_losses.sum() * args_exp.sat_scale
              if init is not None and args.init_scale:
                  init_losses = lpips_model(x_in, init)
                  loss = loss + init_losses.sum() * args.init_scale
              x_in_grad += torch.autograd.grad(loss, x_in)[0]
              if torch.isnan(x_in_grad).any()==False:
                  grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
              else:
                # print("NaN'd")
                x_is_NaN = True
                grad = torch.zeros_like(x)
          if args.clamp_grad and x_is_NaN == False:
              magnitude = grad.square().mean().sqrt()
              return grad * magnitude.clamp(max=args.clamp_max) / magnitude  #min=-0.02, min=-clamp_max, 
          return grad
  
      if args.diffusion_sampling_mode == 'ddim':
          sample_fn = diffusion.ddim_sample_loop_progressive
      else:
          sample_fn = diffusion.plms_sample_loop_progressive


      image_display = Output()
      for i in range(args.n_batches):
          if args.animation_mode == 'None':
            display.clear_output(wait=True)
            batchBar = tqdm(range(args.n_batches), desc ="Batches")
            batchBar.n = i
            batchBar.refresh()
          print('')
          display.display(image_display)
          gc.collect()
          torch.cuda.empty_cache()
          cur_t = diffusion.num_timesteps - skip_steps - 1
          total_steps = cur_t

          if args_exp.perlin_init:
              init = regen_perlin()

          if args.diffusion_sampling_mode == 'ddim':
              samples = sample_fn(
                  model,
                  (args_exp.batch_size, 3, args.side_y, args.side_x),
                  clip_denoised=args_exp.clip_denoised,
                  model_kwargs={},
                  cond_fn=cond_fn,
                  progress=True,
                  skip_timesteps=skip_steps,
                  init_image=init,
                  randomize_class=args_exp.randomize_class,
                  eta=args_exp.eta,
              )
          else:
              samples = sample_fn(
                  model,
                  (args_exp.batch_size, 3, args.side_y, args.side_x),
                  clip_denoised=args_exp.clip_denoised,
                  model_kwargs={},
                  cond_fn=cond_fn,
                  progress=True,
                  skip_timesteps=skip_steps,
                  init_image=init,
                  randomize_class=args_exp.randomize_class,
                  order=2,
              )
          
          
          # with run_display:
          # display.clear_output(wait=True)
          for j, sample in enumerate(samples):    
            cur_t -= 1
            intermediateStep = False
            if args.steps_per_checkpoint is not None:
                if j % args_exp.steps_per_checkpoint == 0 and j > 0:
                  intermediateStep = True
            elif j in args.intermediate_saves:
              intermediateStep = True
            with image_display:
              if j % args.display_rate == 0 or cur_t == -1 or intermediateStep == True:
                  for k, image in enumerate(sample['pred_xstart']):
                      # tqdm.write(f'Batch {i}, step {j}, output {k}:')
                      current_time = datetime.now().strftime('%y%m%d-%H%M%S_%f')
                      percent = math.ceil(j/total_steps*100)
                      if args.n_batches > 0:
                        #if intermediates are saved to the subfolder, don't append a step or percentage to the name
                        if cur_t == -1 and args.intermediates_in_subfolder is True:
                          save_num = f'{frame_num:04}' if args_exp.animation_mode != "None" else i
                          filename = f'{args.batch_name}({args.batchNum})_{save_num}.png'
                        else:
                          #If we're working with percentages, append it
                          if args.steps_per_checkpoint is not None:
                            filename = f'{args.batch_name}({args.batchNum})_{i:04}-{percent:02}%.png'
                          # Or else, iIf we're working with specific steps, append those
                          else:
                            filename = f'{args.batch_name}({args.batchNum})_{i:04}-{j:03}.png'
                      image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                      if j % args.display_rate == 0 or cur_t == -1:
                        image.save('progress.png')
                        display.clear_output(wait=True)
                        display.display(display.Image('progress.png'))
                      if args.steps_per_checkpoint is not None:
                        if j % args.steps_per_checkpoint == 0 and j > 0:
                          if args.intermediates_in_subfolder is True:
                            image.save(f'{args_exp.partialFolder}/{filename}')
                          else:
                            image.save(f'{args_exp.batchFolder}/{filename}')
                      else:
                        if j in args.intermediate_saves:
                          if args.intermediates_in_subfolder is True:
                            image.save(f'{args_exp.partialFolder}/{filename}')
                          else:
                            image.save(f'{args_exp.batchFolder}/{filename}')
                      if cur_t == -1:
                        if frame_num == 0:
                          save_settings(args_exp)
                        if args.animation_mode != "None":
                          image.save('prevFrame.png')
                        image.save(f'{args_exp.batchFolder}/{filename}')
                        if args.animation_mode == "3D":
                          # If turbo, save a blended image
                          if args_exp.turbo_mode and frame_num > 0:
                            # Mix new image with prevFrameScaled
                            blend_factor = (1)/int(args_exp.turbo_steps)
                            newFrame = cv2.imread('prevFrame.png') # This is already updated..
                            prev_frame_warped = cv2.imread('prevFrameScaled.png')
                            blendedImage = cv2.addWeighted(newFrame, blend_factor, prev_frame_warped, (1-blend_factor), 0.0)
                            cv2.imwrite(f'{args_exp.batchFolder}/{filename}',blendedImage)
                          else:
                            image.save(f'{args_exp.batchFolder}/{filename}')

                          if args_exp.vr_mode:
                            generate_eye_views(TRANSLATION_SCALE, args_exp.batchFolder, filename, frame_num, midas_model, midas_transform, args, args_exp)

                        # if frame_num != args.max_frames-1:
                        #   display.clear_output()
          
          plt.plot(np.array(loss_values), 'r')


def generate_eye_views(trans_scale,batchFolder,filename,frame_num,midas_model, midas_transform, args, args_exp):
   for i in range(2):
      theta = args_exp.vr_eye_angle * (math.pi/180)
      ray_origin = math.cos(theta) * args_exp.vr_ipd / 2 * (-1.0 if i==0 else 1.0)
      ray_rotation = (theta if i==0 else -theta)
      translate_xyz = [-(ray_origin)*trans_scale, 0,0]
      rotate_xyz = [0, (ray_rotation), 0]
      rot_mat = p3dT.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=args_exp.device), "XYZ").unsqueeze(0)
      transformed_image = dxf.transform_image_3d(f'{batchFolder}/{filename}', midas_model, midas_transform, args_exp.device,
                                                      rot_mat, translate_xyz, args.near_plane, args.far_plane,
                                                      args.fov, padding_mode=args.padding_mode,
                                                      sampling_mode=args.sampling_mode, midas_weight=args.midas_weight,spherical=True)
      eye_file_path = batchFolder+f"/frame_{frame_num:04}" + ('_l' if i==0 else '_r')+'.png'
      transformed_image.save(eye_file_path)

def save_settings(args):
  setting_list = {
    'text_prompts': args.text_prompts,
    'image_prompts': args.image_prompts,
    'clip_guidance_scale': args.clip_guidance_scale,
    'tv_scale': args.tv_scale,
    'range_scale': args.range_scale,
    'sat_scale': args.sat_scale,
    # 'cutn': cutn,
    'cutn_batches': args.cutn_batches,
    'max_frames': args.max_frames,
    'interp_spline': args.interp_spline,
    # 'rotation_per_frame': rotation_per_frame,
    'init_image': args.init_image,
    'init_scale': args.init_scale,
    'skip_steps': args.skip_steps,
    # 'zoom_per_frame': zoom_per_frame,
    'frames_scale': args.frames_scale,
    'frames_skip_steps': args.frames_skip_steps,
    'perlin_init': args.perlin_init,
    'perlin_mode': args.perlin_mode,
    'skip_augs': args.skip_augs,
    'randomize_class': args.randomize_class,
    'clip_denoised': args.clip_denoised,
    'clamp_grad': args.clamp_grad,
    'clamp_max': args.clamp_max,
    'seed': args.seed,
    'fuzzy_prompt': args.fuzzy_prompt,
    'rand_mag': args.rand_mag,
    'eta': args.eta,
    'width': args.width_height[0],
    'height': args.width_height[1],
    'diffusion_model': args.diffusion_model,
    'use_secondary_model': args.use_secondary_model,
    'steps': args.steps,
    'diffusion_steps': args.diffusion_steps,
    'diffusion_sampling_mode': args.diffusion_sampling_mode,
    'ViTB32': args.ViTB32,
    'ViTB16': args.ViTB16,
    'ViTL14': args.ViTL14,
    'RN101': args.RN101,
    'RN50': args.RN50,
    'RN50x4': args.RN50x4,
    'RN50x16': args.RN50x16,
    'RN50x64': args.RN50x64,
    'cut_overview': str(args.cut_overview),
    'cut_innercut': str(args.cut_innercut),
    'cut_ic_pow': args.cut_ic_pow,
    'cut_icgray_p': str(args.cut_icgray_p),
    'key_frames': args.key_frames,
    'max_frames': args.max_frames,
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
    'video_init_path':args.video_init_path,
    'extract_nth_frame':args.extract_nth_frame,
    'video_init_seed_continuity': args.video_init_seed_continuity,
    'turbo_mode':args.turbo_mode,
    'turbo_steps':args.turbo_steps,
    'turbo_preroll':args.turbo_preroll,
  }
  # print('Settings:', setting_list)
  with open(f"{args.batchFolder}/{args.batch_name}({args.batchNum})_settings.txt", "w+") as f:   #save settings
    json.dump(setting_list, f, ensure_ascii=False, indent=4)
