import torch 
import torch.nn as nn



class Discriminator(nn.Module):
    
    def __init__(self, in_channels = 3,
                 conv_channels = [64, 128, 256],
                 kernels = [4, 4, 4, 4],
                 strides = [2, 2, 2, 1],
                 paddings = [1, 1, 1, 1]):
        
        super().__init__()
        self.in_channels = in_channels
        activation = nn.LeakyReLU(0.2)
        layers_dim = [in_channels] + conv_channels + [1]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels = layers_dim[i],
                          out_channels = layers_dim[i+1],
                          kernel_size = kernels[i],
                          stride = strides[i],
                          padding = paddings[i]),
                nn.BatchNorm2d(layers_dim[i+1]) if i < len(layers_dim) - 2 else nn.Identity(),
                activation if i < len(layers_dim) - 2 else nn.Sigmoid()
            )
            
            for i in range(len(layers_dim) - 1)
        ])
        
        
        
    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)
        
        return x
    



        