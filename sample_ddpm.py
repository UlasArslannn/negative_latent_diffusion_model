import torch
import torch.nn as nn
from tqdm import tqdm



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def sample(model, scheduler, train_config, diffusion_config, model_config):
    
    xt = torch.randn(train_config['num_samples'],
                             model_config['in_channels'],
                             model_config['image_size'],
                             model_config['image_size']).to(device)
    
    model.eval()
    
    for i in tqdm(reversed(range(scheduler.num_train_timesteps))):
        
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        
        
        xt , xt_1_pred = scheduler.step(noise_pred, torch.as_tensor(i).to(device), xt)
        
        ims = torch.clamp(xt, -1, 1.).detach().cpu()
        ims = (ims + 1) / 2.0
        
        grid = make_grid(ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)
        if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
            os.mkdir(os.path.join(train_config['task_name'], 'samples'))
        img.save(os.path.join(train_config['task_name'], 'samples', 'x0_{}.png'.format(i)))
        img.close()
        
    
    
    
def infer(args):
    
    with open(args.config_path, 'r') as f:
        
        config = yaml.safe_load(f)
        
    print("Configuration:", config)
    
    model_config = config['model_params']
    training_config = config['training_params']
    diffusion_config = config['diffusion_params']
    
    
    noise_Scheduler = DDPMSchedulercheckpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],train_config['ckpt_name']),map_location=device))
    
    sample(model, noise_Scheduler, training_config, diffusion_config, model_config)
    
    model.eval()
    
    with torch.no_grad():
        sample(model, noise_Scheduler, training_config, diffusion_config, model_config)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='congig_path', type=str, default = 'config/default.yaml',help='Path to config file')

    args = parser.parse_args()
    infer(args)    
    #command
        
        
        
        