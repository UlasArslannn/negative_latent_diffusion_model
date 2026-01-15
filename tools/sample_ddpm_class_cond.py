import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, diffusion_model_config,
           autoencoder_model_config, diffusion_config, dataset_config, vae,
           target_class=None, avoid_list=None):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    
    Args:
        target_class: int or None. If None, generates unconditionally
        avoid_list: list of ints. Classes to avoid during generation
    """
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    
    ########### Sample random noise latent ##########
    xt = torch.randn((train_config['num_samples'],
                      autoencoder_model_config['z_channels'],
                      im_size,
                      im_size)).to(device)
    ###############################################
    
    ############# Validate the config #################
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    assert condition_config is not None, ("This sampling script is for class conditional "
                                          "but no conditioning config found")
    condition_types = get_config_value(condition_config, 'condition_types', [])
    assert 'class' in condition_types, ("This sampling script is for class conditional "
                                          "but no class condition found in config")
    validate_class_config(condition_config)
    ###############################################
    
    ############ Create Conditional input ###############
    num_classes = condition_config['class_condition_config']['num_classes']
    batch_size = train_config['num_samples']
    
    # Handle target class
    if target_class is not None:
        # Specific class conditioning
        sample_classes = torch.tensor([target_class] * batch_size)
        print(f'Generating {batch_size} images for class {target_class}')
    else:
        # Random class conditioning - exclude avoid_list classes
        available_classes = [c for c in range(num_classes) if c not in (avoid_list or [])]
        if len(available_classes) == 0:
            raise ValueError("No classes available after excluding avoid_list!")
        sample_classes = torch.tensor([available_classes[i % len(available_classes)] 
                                        for i in torch.randint(0, len(available_classes), (batch_size,)).tolist()])
        print('Generating images for {}'.format(list(sample_classes.numpy())))
    
    cond_input = {
        'class': torch.nn.functional.one_hot(sample_classes, num_classes).float().to(device)
    }
    
    # Handle avoid list
    if avoid_list and len(avoid_list) > 0:
        print(f'Avoiding classes: {avoid_list}')
        avoid_tensor = torch.zeros((batch_size, num_classes), device=device)
        for cls in avoid_list:
            avoid_tensor[:, cls] = 1.0
        cond_input['avoid'] = avoid_tensor
    else:
        # Empty avoid condition
        cond_input['avoid'] = torch.zeros((batch_size, num_classes), device=device)
    
    # Unconditional input for classifier free guidance
    uncond_input = {
        'class': cond_input['class'] * 0,
        'avoid': torch.zeros((batch_size, num_classes), device=device)
    }
    ###############################################
    
    # By default classifier free guidance is disabled
    # Change value in config or change default value here to enable it
    cf_guidance_scale = get_config_value(train_config, 'cf_guidance_scale', 1.0)
    
    ################# Sampling Loop ########################
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        t = (torch.ones((xt.shape[0],))*i).long().to(device)
        noise_pred_cond = model(xt, t, cond_input)
        
        if cf_guidance_scale > 1:
            noise_pred_uncond = model(xt, t, uncond_input)
            noise_pred = noise_pred_uncond + cf_guidance_scale*(noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
        if i == 0:
            # Decode ONLY the final image to save time
            ims = vae.decode(xt)
        else:
            ims = x0_pred
        
        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        
        # Convert each image to PIL and add class labels
        im_size = ims.shape[-1]
        scale_factor = 2  # Scale up images for better visibility
        scaled_size = im_size * scale_factor
        label_height = 25
        num_images = ims.shape[0]
        num_cols = 5  # 5 columns grid
        num_rows = (num_images + num_cols - 1) // num_cols  # Ceiling division
        
        # Create a new image with grid layout
        total_width = scaled_size * num_cols
        total_height = (scaled_size + label_height) * num_rows
        combined_img = Image.new('RGB', (total_width, total_height), color='white')
        draw = ImageDraw.Draw(combined_img)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        # Place each image and its label in grid
        for idx in range(num_images):
            single_im = ims[idx]
            single_pil = torchvision.transforms.ToPILImage()(single_im)
            # Scale up the image
            single_pil = single_pil.resize((scaled_size, scaled_size), Image.NEAREST)
            
            # Calculate grid position
            row = idx // num_cols
            col = idx % num_cols
            x_offset = col * scaled_size
            y_offset = row * (scaled_size + label_height)
            
            # Paste image
            combined_img.paste(single_pil, (x_offset, y_offset))
            
            # Draw class label below the image
            target_cls = sample_classes[idx].item()
            label_text = f"cls:{target_cls}"
            
            # Add avoid info if present
            if avoid_list and len(avoid_list) > 0:
                label_text = f"t:{target_cls} a:{avoid_list}"
            
            # Center the text
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = x_offset + (scaled_size - text_width) // 2
            draw.text((text_x, y_offset + scaled_size + 2), label_text, fill='black', font=font)
        
        if not os.path.exists(os.path.join(train_config['task_name'], 'cond_class_samples')):
            os.mkdir(os.path.join(train_config['task_name'], 'cond_class_samples'))
        combined_img.save(os.path.join(train_config['task_name'], 'cond_class_samples', 'x0_{}.png'.format(i)))
        combined_img.close()
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
                                       map_location=device), strict=True)
    else:
        raise Exception('VAE checkpoint {} not found'.format(os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name'])))
    #####################################
    
    # Parse target class
    target_class = args.target_class if args.target_class >= 0 else None
    
    # Parse avoid list
    avoid_list = []
    if args.avoid:
        avoid_list = [int(x.strip()) for x in args.avoid.split(',')]
    
    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae,
               target_class=target_class, avoid_list=avoid_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation for class conditional '
                                                 'Mnist generation with avoid conditioning')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist_class_cond.yaml', type=str)
    parser.add_argument('--class', dest='target_class', type=int, default=-1,
                        help='Target class to generate (0-9 for MNIST). Use -1 for random.')
    parser.add_argument('--avoid', dest='avoid', type=str, default='',
                        help='Comma-separated list of classes to avoid, e.g., "1,9"')
    args = parser.parse_args()
    infer(args)