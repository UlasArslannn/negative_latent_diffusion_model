import torch 
import torch.nn as nn
import yaml
import numpy as np
import os
from torch.optim import Adam
from torch.utils.data import DataLoader
#from dataset.MnistDataset import MnistDataset
from model import UNetModel
from tqdm import tqdm
#import mnist
from diffusers.schedulers import DDPMScheduler
from CelebDataset import CelebDataset
import argparse



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    
    with open(args.config_path, 'r') as f:
        
        config = yaml.safe_load(f)
        
    print("Configuration:", config)
    
    model_config = config['model_params']
    training_config = config['training_params']
    diffusion_config = config['diffusion_params']
    
    base = './archive (6)/Celebrity Faces Dataset'
    
    celeb_dataset = CelebDataset(image_paths=base, transform=True)
    celeb_dataLoader = DataLoader(celeb_dataset, batch_size=training_config['batch_size'], shuffle=True)
    
    noise_Scheduler = DDPMScheduler(num_train_timesteps=diffusion_config['num_timesteps'],
                                    beta_start=diffusion_config['beta_start'],
                                        beta_end=diffusion_config['beta_end'] )
    
    model = UNetModel(model_config=model_config).to(device)
    optimizer = Adam(model.parameters(), lr=float(training_config['learning_rate']))
    
    criterion = torch.nn.MSELoss()
    num_epochs = training_config['num_epochs']
    
    model.train()
    
    for epoch in range(num_epochs):
        
        loss_logger = []
        for step , batch in enumerate(tqdm(celeb_dataLoader)):
            
            batch = batch.float().to(device)
            b_size = batch.shape[0]
            
            timesteps = torch.randint(0, noise_Scheduler.num_train_timesteps, (b_size,), device=device).long()
            noise = torch.randn_like(batch).to(device)
            
            noisy_images = noise_Scheduler.add_noise(batch, noise, timesteps)
            
            predicted_noise = model(noisy_images, timesteps)
            
            loss = criterion(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_logger.append(loss.item())
            
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch + 1,
            np.mean(loss_logger),
        ))
        
        torch.save(model.state_dict(), 'ddpm_celeba_epoch_{}.pth'.format(epoch+1))
        
    print('Done Training ....')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train DDPM model on CelebA dataset')
    
    parser.add_argument('--config_path', type=str, default='./config/default.yaml', help='Path to the config file')
    args = parser.parse_args()
    train(args)


    
    

