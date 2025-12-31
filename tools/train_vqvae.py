import yaml
import argparse
import torch
import random
import torchvision
import os
import numpy as np
from tqdm import tqdm
from models.vqvae import VQVAE
from models.lpips import LPIPS
from models.discriminator import Discriminator
from torch.utils.data.dataloader import DataLoader
from dataset.mnist_dataset import MnistDataset
from dataset.celeb_dataset import CelebDataset
from torch.optim import Adam
from torchvision.utils import make_grid
from torch import nn

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
    training_config = config['train_params']
    
    # set the desired seed value
    seed = training_config['seed']
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    
    
    
    model = VQVAE(dataset_config['im_channels'] , autoencoder_config).to(device)
    
    #create the dataset 
    
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebA': CelebDataset
    }.get(dataset_config['name'])


    train_dataset = im_dataset_cls(split = 'train',
                                   im_path = dataset_config['im_path'],
                                   im_size = dataset_config['im_size'],
                                   im_channels = dataset_config['im_channels']
                                   )    
    
    
    data_loader = DataLoader(train_dataset, 
                             batch_size= training_config['autoencoder_batch_size'],
                             shuffle=True,)
    
    # Creating mnist or CelebA dataset folder
    if not os.path.exists(training_config['task_name']):
        os.mkdir(training_config['task_name'])

    
    num_epochs = training_config['autoencoder_epochs']
    
    
    # Reconstruction loss
    recon_criterion = nn.MSELoss()
    
    # Discriminator los  --> maybe adversarial loss
    
    disc_criterion = nn.MSELoss()
    
    lpips_model = LPIPS().eval().to(device)
    disc_model = Discriminator(dataset_config['im_channels']).to(device)
    
    optimizer_disc = Adam(disc_model.parameters() , lr = training_config['autoencoder_lr'], betas = (0.5, 0.999))
    optimizer_gen = Adam(model.parameters(), lr = training_config['autoencoder_lr'],  betas = (0.5, 0.999))  # Our VQ-VAE is the generator
    
    disc_step_start = training_config['disc_start']
    step_count = 0
    
    
    acc_steps = training_config['autoencoder_acc_steps']
    image_save_steps = training_config['autoencoder_img_save_steps']
    img_save_count = 0
    
    for epoch in range(num_epochs):
        
        recon_loss_logger = []
        codebook_loss_logger = []
        gen_loss_logger = []
        perceptual_loss_logger = []
        losses = []
        disc_loss_logger = []
        
        optimizer_disc.zero_grad()
        optimizer_gen.zero_grad()
        
        for im in tqdm(data_loader):
            
            im = im.float().to(device)
            
            step_count += 1
            
            
            # ========== Train Generator (VQ-VAE) ==========
            
            model_output = model(im)
            recon_im , z, quantize_losses = model_output
            
            
            
            
            # Image Saving Logic
            if step_count % image_save_steps == 0 or step_count == 1:
                sample_size = min(8, im.shape[0])
                save_output = torch.clamp(recon_im[:sample_size], -1., 1.).detach().cpu()
                save_output = ((save_output + 1) / 2)
                save_input = ((im[:sample_size] + 1) / 2).detach().cpu()
                
                grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
                img = torchvision.transforms.ToPILImage()(grid)
                if not os.path.exists(os.path.join(training_config['task_name'],'vqvae_autoencoder_samples')):
                    os.mkdir(os.path.join(training_config['task_name'], 'vqvae_autoencoder_samples'))
                img.save(os.path.join(training_config['task_name'],'vqvae_autoencoder_samples',
                                      'current_autoencoder_sample_{}.png'.format(img_save_count)))
                img_save_count += 1
                img.close()
                
            # L2 Loss
            recon_loss = recon_criterion(recon_im, im)
            recon_loss_logger.append(recon_loss.item())
            recon_loss = recon_loss / acc_steps
            
            # Addition losses to recon loss
            codebook_loss = training_config['codebook_weight'] * quantize_losses['codebook_loss'] / acc_steps
            commitment_loss = training_config['commitment_beta'] * quantize_losses['commitment_loss'] / acc_steps
            
            g_loss = recon_loss + codebook_loss + commitment_loss 
            
            codebook_loss_logger.append(recon_loss.item())
            
            # Adversarial Loss but if disc step has started
            if step_count > disc_step_start:
                
                disc_fake_pred = disc_model(recon_im)
                disc_fake_loss = disc_criterion(disc_fake_pred, 
                                                torch.ones_like(disc_fake_pred,
                                                                device = disc_fake_pred.device))
                
                gen_loss_logger.append(training_config['disc_weight'] * disc_fake_loss.item())
                g_loss += training_config['disc_weight'] * disc_fake_loss / acc_steps
            
            lpips_loss = torch.mean(lpips_model(recon_im, im))
            perceptual_loss_logger.append(training_config['perceptual_weight'] * lpips_loss.item())
            g_loss += training_config['perceptual_weight'] * lpips_loss / acc_steps
            losses.append(g_loss.item())
            g_loss.backward()
            
            # ========== Optimize Discriminator ==========
            

            if step_count > disc_step_start:
                fake  = recon_im
                disc_fake_pred = disc_model(fake.detach())
                disc_real_pred = disc_model(im)
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.zeros_like(disc_fake_pred,
                                                                 device = disc_fake_pred.device))
                
                disc_real_loss = disc_criterion(disc_real_pred,
                                                torch.ones_like(disc_real_pred,
                                                                device = disc_real_pred.device))
                
                disc_loss = training_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
                disc_loss_logger.append(disc_loss.item())
                disc_loss = disc_loss / acc_steps
                disc_loss.backward()
                if step_count % acc_steps == 0:
                    optimizer_disc.step()
                    optimizer_disc.zero_grad()
                    
            if step_count % acc_steps == 0:
                optimizer_gen.step()
                optimizer_gen.zero_grad()
            
        optimizer_disc.step()
        optimizer_disc.zero_grad()
        optimizer_gen.step()
        optimizer_gen.zero_grad()
        
        if len(disc_loss_logger) > 0:
            print(
                'Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | '
                'Codebook : {:.4f} | G Loss : {:.4f} | D Loss {:.4f}'.
                format(epoch + 1,
                       np.mean(recon_loss_logger),
                       np.mean(perceptual_loss_logger),
                       np.mean(codebook_loss_logger),
                       np.mean(gen_loss_logger),
                       np.mean(disc_loss_logger)))
        else:
            print('Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | Codebook : {:.4f}'.
                  format(epoch + 1,
                         np.mean(recon_loss_logger),
                         np.mean(perceptual_loss_logger),
                         np.mean(disc_loss_logger)))

        torch.save(model.state_dict(), os.path.join(training_config['task_name'],
                                                    training_config['vqvae_autoencoder_ckpt_name']))
        torch.save(disc_model.state_dict(), os.path.join(training_config['task_name'],
                                                            training_config['vqvae_discriminator_ckpt_name']))
    print('Done Training...')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path',
                        default='./config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args)