import argparse
import os
from utils.utils import str2bool, download_models, do_run, create_dirs, setting_device, correct_sizes, create_list_clip_models, set_intermediate_saves
import torch
import gc
from glob import glob
from types import SimpleNamespace
import warnings
import sys
import pathlib
import subprocess
import math
import random

# Add 3rd-party methods
sys.path.append('./AdaBins')
sys.path.append('./disco_difussion')
sys.path.append('./guided_diffusion')
sys.path.append('./MiDaS')
sys.path.append('./pytorch3d-lite') 
sys.path.append('./ResizeRight') 

from guided_diffusion.script_util import create_model_and_diffusion, init_model_configs
from secondary_diffusion_model import SecondaryDiffusionImageNet2
from clip import clip
from animation_utils import parse_key_frames, split_prompts, get_inbetweens
MAX_ADABINS_AREA = 500000


def parse_args():
    desc = "DiscoDiffusion"

    parser = argparse.ArgumentParser(description=desc)

    # Dataset parameters and input paths
    parser.add_argument('--root_path', type=str, default="pwd", help='Path root where inputs and outputs are located. By default is cwd().')
    parser.add_argument('--init_images_path', type=str, default="init_images", help='Folder name for init images')
    parser.add_argument('--images_out_path', type=str, default="images_out", help='Folder name for images out')
    parser.add_argument('--model_path', type=str, default="models", help='Folder name for models')
    parser.add_argument('--pretrained_path', type=str, default="pretrained", help='Folder name for pretrained')
    parser.add_argument('--videoFramesFolder', type=str, default="videoFrames", help='Folder name for videoFrames')

    # Experiment parameters
    parser.add_argument('--experiment_name', type=str, default="", help='A name for the experiment')

    # Warnings parameters
    parser.add_argument('--warnings', type=str2bool, nargs='?', const=False, default=True, help="Show warnings")

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

    return parser.parse_args()


def main():
    args = parse_args()

    # warnings
    if args.warnings:
        warnings.filterwarnings("ignore")

    # Directories config
    create_dirs(args)

    # Import device
    setting_device(args)
    
    # Download models
    download_models(args)

    # Secondary model
    if args.use_secondary_model:
        secondary_model = SecondaryDiffusionImageNet2()
        secondary_model.load_state_dict(torch.load(f'{args.model_path}/secondary_model_imagenet_2.pth', map_location='cpu'))
        secondary_model.eval().requires_grad_(False).to(args.device)

    # Create list of clips models
    create_list_clip_models(args)
        
    #Get corrected sizes
    args.side_x, args.side_y = correct_sizes(args.width, args.heigth)
    
    # Saving
    set_intermediate_saves(args)

    # Prompt
    args.text_prompts = {
        0: [args.text_prompts],
        100: [args.text_prompts_100_0, args.text_prompts_100_1],
    }

    if args.image_prompts == "None":
        args.image_prompts = {}
    else:
        args.image_prompts = {
            args.image_prompts,
        }

    # Do the run!!
    # Model config
    args.model_config = init_model_configs(args)
    args.batch_size = 1 
    print('llego hasta acá')
    def move_files(start_num, end_num, old_folder, new_folder):
        for i in range(start_num, end_num):
            old_file = old_folder + f'/{args.batch_name}({args.batchNum})_{i:04}.png'
            new_file = new_folder + f'/{args.batch_name}({args.batchNum})_{i:04}.png'
            os.rename(old_file, new_file)

    # if args.retain_overwritten_frames is True:
    #     args.retainFolder = f'{args.batchFolder}/retained'
    #     os.makedirs(args.retainFolder, exist_ok=True)


    # args.skip_step_ratio = int(args.frames_skip_steps.rstrip("%")) / 100
    # args.calc_frames_skip_steps = math.floor(args.steps * args.skip_step_ratio)

    # if args.steps <= args.calc_frames_skip_steps:
    #     sys.exit("ERROR: You can't skip more steps than your total steps")

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
            # if args.animation_mode != '3D' and args.turbo_mode == True and args.start_frame > args.turbo_preroll and args.start_frame % int(args.turbo_steps) != 0:
            #     args.start_frame = args.start_frame - (args.start_frame % int(args.turbo_steps))
        else:
            args.start_frame = int(args.resume_from_frame)+1
            # if args.animation_mode != '3D' and args.turbo_mode == True and args.start_frame > args.turbo_preroll and args.start_frame % int(args.turbo_steps) != 0:
            #     args.start_frame = args.start_frame - (args.start_frame % int(args.turbo_steps))
            # if args.retain_overwritten_frames is True:
            #     args.existing_frames = len(glob(args.batchFolder+f"/{args.batch_name}({args.batchNum})_*.png"))
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
        'steps_per_checkpoint': args.steps_per_checkpoint,
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

    args_m = SimpleNamespace(**args_m)

    print('Prepping model...')
    model, diffusion = create_model_and_diffusion(**args.model_config)
    model.load_state_dict(torch.load(f'{args.model_path}/{args.diffusion_model}.pt', map_location='cpu'))
    model.requires_grad_(False).eval().to(args.device)
    for name, param in model.named_parameters():
        if 'qkv' in name or 'norm' in name or 'proj' in name:
            param.requires_grad_()
    if args.model_config['use_fp16']:
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