import numpy as np
import torch
import random
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from transformers import DistilBertModel, DistilBertTokenizer, CLIPTokenizer, CLIPTextModel
from utils.config_utils import *
from utils.text_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, diffusion_model_config,
           autoencoder_model_config, diffusion_config, dataset_config, vae, text_tokenizer, text_model,
           text_prompt, negative_prompt=None, cfg_scale_override=None):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    
    ########### Sample random noise latent ##########
    # For not fixing generation with one sample
    xt = torch.randn((1,
                      autoencoder_model_config['z_channels'],
                      im_size,
                      im_size)).to(device)
    ###############################################
    
    ############ Create Conditional input ###############
    # Get text embeddings for the prompt
    text_prompt_list = [text_prompt]
    text_prompt_embed = get_text_representation(text_prompt_list,
                                                text_tokenizer,
                                                text_model,
                                                device)
    
    # Use negative prompt if provided, otherwise use empty prompt
    if negative_prompt and len(negative_prompt.strip()) > 0:
        neg_prompt_list = [negative_prompt]
        uncond_text_embed = get_text_representation(neg_prompt_list, text_tokenizer, text_model, device)
        print(f"Using NEGATIVE prompt: '{negative_prompt}'")
    else:
        empty_prompt = ['']
        uncond_text_embed = get_text_representation(empty_prompt, text_tokenizer, text_model, device)
        print("Using empty prompt (no negative conditioning)")
    
    assert uncond_text_embed.shape == text_prompt_embed.shape
    print(f"Generating with prompt: '{text_prompt}'")
    
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    
    # Check if image conditioning is used
    use_image_cond = 'image' in condition_config.get('condition_types', []) if condition_config else False
    
    if use_image_cond:
        validate_image_config(condition_config)
        # Generate a random mask instead of loading from dataset
        # This removes the need for the dataset during sampling
        mask_channels = condition_config['image_condition_config']['image_condition_input_channels']
        mask_h = condition_config['image_condition_config']['image_condition_h']
        mask_w = condition_config['image_condition_config']['image_condition_w']
        
        # Create a random segmentation-like mask (values 0 or 1 per channel)
        # Each channel represents a face part (skin, nose, eyes, etc.)
        mask = torch.zeros((1, mask_channels, mask_h, mask_w), device=device)
        # Randomly activate some mask channels to simulate face parts
        for ch in range(mask_channels):
            if random.random() > 0.3:  # 70% chance to include each part
                # Create a random elliptical region for this face part
                center_y, center_x = random.randint(mask_h//4, 3*mask_h//4), random.randint(mask_w//4, 3*mask_w//4)
                radius_y, radius_x = random.randint(mask_h//8, mask_h//3), random.randint(mask_w//8, mask_w//3)
                y_grid, x_grid = torch.meshgrid(torch.arange(mask_h), torch.arange(mask_w), indexing='ij')
                ellipse = ((y_grid - center_y)**2 / (radius_y**2 + 1e-6) + (x_grid - center_x)**2 / (radius_x**2 + 1e-6)) < 1
                mask[0, ch] = ellipse.float().to(device)
        print(f"Generated random mask with shape: {mask.shape}")
    else:
        mask = None
        print("No image conditioning used")
    
    # Build conditional inputs
    if mask is not None:
        uncond_input = {
            'text': uncond_text_embed,
            'image': torch.zeros_like(mask)
        }
        cond_input = {
            'text': text_prompt_embed,
            'image': mask
        }
    else:
        # Text-only conditioning
        uncond_input = {
            'text': uncond_text_embed
        }
        cond_input = {
            'text': text_prompt_embed
        }
    
    # Get CFG scale - command line override takes priority
    cf_guidance_scale = cfg_scale_override if cfg_scale_override is not None else get_config_value(train_config, 'cf_guidance_scale', 1.0)
    if cf_guidance_scale > 1:
        print(f"CFG enabled with scale: {cf_guidance_scale}")
    else:
        print("CFG disabled (scale <= 1). Use --cfg_scale 7.5 or set cf_guidance_scale > 1 in config for negative prompting to work.")
    ###############################################
    
    # CFG scale already retrieved above
    # cf_guidance_scale > 1 enables classifier-free guidance with negative prompting
    
    ################# Sampling Loop ########################
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        t = (torch.ones((xt.shape[0],)) * i).long().to(device)
        noise_pred_cond = model(xt, t, cond_input)
        
        if cf_guidance_scale > 1:
            noise_pred_uncond = model(xt, t, uncond_input)
            noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
        # Save x0
        if i == 0:
            # Decode ONLY the final image to save time
            ims = vae.decode(xt)
        else:
            ims = x0_pred
        
        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=10)
        img = torchvision.transforms.ToPILImage()(grid)
        
        if not os.path.exists(os.path.join(train_config['task_name'], 'cond_text_image_samples')):
            os.mkdir(os.path.join(train_config['task_name'], 'cond_text_image_samples'))
        img.save(os.path.join(train_config['task_name'], 'cond_text_image_samples', 'x0_{}.png'.format(i)))
        img.close()
    ##############################################################

def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    
    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    ###############################################
    
    ############# Validate the config #################
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    assert condition_config is not None, ("This sampling script is for image and text conditional "
                                          "but no conditioning config found")
    condition_types = get_config_value(condition_config, 'condition_types', [])
    assert 'text' in condition_types, ("This sampling script is for image and text conditional "
                                       "but no text condition found in config")
    assert 'image' in condition_types, ("This sampling script is for image and text conditional "
                                       "but no image condition found in config")
    validate_text_config(condition_config)
    validate_image_config(condition_config)
    ###############################################
    
    ############# Load tokenizer and text model #################
    with torch.no_grad():
        # Load tokenizer and text model based on config
        # Also get empty text representation
        text_tokenizer, text_model = get_tokenizer_and_model(condition_config['text_condition_config']
                                                             ['text_embed_model'], device=device)
    ###############################################
    
    ########## Load Unet #############
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.eval()
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['ldm_ckpt_name'])):
        print('Loaded unet checkpoint')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ldm_ckpt_name']),
                                         map_location=device))
    else:
        raise Exception('Model checkpoint {} not found'.format(os.path.join(train_config['task_name'],
                                                                   train_config['ldm_ckpt_name'])))
    #####################################
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    ########## Load VQVAE #############
    vae = VQVAE(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_model_config).to(device)
    vae.eval()
    
    # Load vae if found
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['vqvae_autoencoder_ckpt_name'])):
        print('Loaded vae checkpoint')
        vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name']),
                                       map_location=device))
    else:
        raise Exception('VAE checkpoint {} not found'.format(os.path.join(train_config['task_name'],
                                                                          train_config['vqvae_autoencoder_ckpt_name'])))
    #####################################
    
    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae, text_tokenizer, text_model,
               text_prompt=args.prompt, negative_prompt=args.negative, cfg_scale_override=args.cfg_scale)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation '
                                                 'with text and mask conditioning')
    parser.add_argument('--config', dest='config_path',
                        default='config/celebhq_text_image_cond.yaml', type=str)
    parser.add_argument('--prompt', type=str, 
                        default='A woman with blonde hair wearing lipstick.',
                        help='Text prompt for image generation')
    parser.add_argument('--negative', type=str, default='',
                        help='Negative prompt - features to avoid (requires cf_guidance_scale > 1 in config)')
    parser.add_argument('--cfg_scale', type=float, default=None,
                        help='Override CFG scale from config (use > 1 to enable negative prompting)')
    args = parser.parse_args()
    infer(args)
