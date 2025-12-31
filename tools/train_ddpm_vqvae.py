import torch
import torch.nn as nn
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from models.unet_base import Unet
from models.vqvae import VQVAE
from dataset.mnist_dataset import MnistDataset
from dataset.celeb_dataset import CelebDataset
from torch.utils.data.dataloader import DataLoader
from scheduler.linear_noise_scheduler import LinearNoiseScheduler



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    
    with open(args.config_path, 'r') as f:
        
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    
    print(config)
    
    
    
    # define the parameters from config file
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    diffusion_config = config['diffusion_params']
    train_config = config['train_params']
    diffusion_model_config = config['ldm_params']
    
    
    # set the desired seed value
    seed = train_config['seed']
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        
        
    # create the dataset
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebA': CelebDataset
    }.get(dataset_config['name'])
    
    
    im_dataset = im_dataset_cls(
        split='train',
        im_path=dataset_config['im_path'],
        im_size=dataset_config['im_size'],
        im_channels=dataset_config['im_channels'],
        use_latents=True,
        latent_path=os.path.join(train_config['task_name'],
                                    train_config['vqvae_latent_dir_name'])
    )
    
    data_loader = DataLoader(im_dataset, 
                             batch_size = train_config['ldm_batch_size'],
                             shuffle = True,)
    
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    
    
    model = Unet(
        im_channels = autoencoder_model_config['z_channels'],
        model_config = diffusion_model_config
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    num_epochs = train_config['ldm_epochs']
    criterion = nn.MSELoss()
    
    if not im_dataset.use_latents:
        print('Loading vqvae model as latents not present')
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
            
    if not im_dataset.use_latents:
        for param in vae.parameters():
            param.requires_grad = False
    
    
    model.train()
    
    
    for epoch in range(num_epochs):
        
        losses = []
        
        for im in tqdm(data_loader):
            
            im = im.float().to(device)
            
            if not im_dataset.use_latents:
                with torch.no_grad():
                    latents, _= vae.encode(im)
            
            
            noise = torch.rand_like(im).to(device)
            
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            noisy_image = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_image, t)
            
            loss = criterion(noise_pred, noise)
            
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, np.mean(losses)))
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                train_config['ldm_ckpt_name']))
        
    print('Training complete')
    

if __name__ == '__main__':    
            
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args)        
        
    