import torch 
import torch.nn as nn


def get_time_embedding(timesteps, embedding_dim):
    """
    Generate sinusoidal time embeddings.

    Args:
        timesteps (torch.Tensor): A tensor of shape (batch_size,) containing the time steps.
        embedding_dim (int): The desired dimension of the time embeddings.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, embedding_dim) containing the time embeddings.
    """
    # 10000^(2i/dim)
    half_dim = embedding_dim // 2
    factor = 10000 ** (torch.arange(half_dim, dtype = torch.float32) / half_dim)
    
    # timesteps: (batch_size,) -> (batch_size, 1)
    # pos / factor

    t_emb = timesteps[:,None].repeat(1, half_dim) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim = -1)
    return t_emb




class DownsampleBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, downsample, t_emb_dim,
                 num_layers = 1, num_heads = 4):
        super().__init__()
        self.downsample = downsample
        self.num_layers = num_layers
        
        self.first_resblock = nn.ModuleList(
            [
                nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 
                        kernel_size = 3, padding = 1, stride = 1)
                
                )
                for i in range(num_layers)
            ]
        )
        
        self.t_emb_layer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ]
        )
        
        self.second_resblock = nn.ModuleList(
            [
                nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, stride = 1)
                )
                for _ in range(num_layers)   
            
            ]
        )
        
        self.res_conv_block = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels , out_channels, kernel_size = 1)
                for i in range(num_layers)
            ]
        )
        
        self.attention_norm = nn.ModuleList(
            [
                nn.GroupNorm(8, out_channels)
                for _ in range(num_layers)
            ]
        )
        
        self.attention = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim = out_channels, num_heads = num_heads, 
                                      batch_first = True)
                for _ in range(num_layers)
            ]
        )
        
        self.downsample_conv = nn.Conv2d(out_channels, out_channels, kernel_size = 4, stride = 2, padding = 1) if self.downsample else nn.Identity()
        
    def forward(self, x, t_emb):
        
        
        for i in range(self.num_layers):
            
            res = x    
            # First Resblock
            x = self.first_resblock[i](x)
            # Time embedding
            t_emb_out = self.t_emb_layer[i](t_emb)[:,:,None,None]
            x = x + t_emb_out
            # Second Resblock
            x = self.second_resblock[i](x)
            # Residual connection
            res = self.res_conv_block[i](res)
            x = x + res
            
            # Attention
            b, c, h, w = x.shape
            x_reshaped = x.reshape(b, c, h * w)
            in_attn = self.attention_norm[i](x_reshaped)
            in_attn = in_attn.permute(0,2,1)
            out_attn, _ = self.attention[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.permute(0,2,1).reshape(b, c, h, w)
            x = x + out_attn
            
        x = self.downsample_conv(x)
        return x 
    
    
class MidBlock(nn.Module):
    
    def __init__(self, in_channels,out_channels, t_emb_dim, num_heads = 4, num_layers = 1 ):
        
        super().__init__()
        
        self.num_layers = num_layers
        
        self.first_resblock = nn.ModuleList(
            [
                nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 
                        kernel_size = 3, padding = 1, stride = 1)
                
                )
                for i in range(num_layers)
            ]
        )
        
        self.t_emb_layer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
            ]
        )
            
        self.second_resblock = nn.ModuleList(
            [
                nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, stride = 1)
                )
                for _ in range(num_layers)   
            
            ]
        )
            
            
        self.attention_norm = nn.ModuleList(
            [
                nn.GroupNorm(8, out_channels)
                for _ in range(num_layers)
            ]
        )
        
        self.attention = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim = out_channels, num_heads = num_heads, 
                                      batch_first = True)
                for i in range(num_layers)
            ]
        )
        
        self.res_conv_block = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels , out_channels, kernel_size = 1)
                for i in range(num_layers)
            ]
        )
            
            
    def forward(self, x, t_emb):
        
        
        res = x 
        
        # First Resblock
        x = self.first_resblock[0](x)
        # Time embedding
        t_emb_out = self.t_emb_layer[0](t_emb)[:,:,None,None]
        x = x + t_emb_out
        # Second Resblock
        x = self.second_resblock[0](x)
        # Residual connection
        res = self.res_conv_block[0](res)
        x = x + res
        
        for i in range(self.num_layers- 1):
            
            
            b, c, h, w = x.shape
            
            in_attn = x.reshape(b, c, h * w)
            in_attn = self.attention_norm[i](in_attn)
            in_attn = in_attn.permute(0,2,1)
            out_attn, _ = self.attention[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.permute(0,2,1).reshape(b, c, h, w)  
            x = x + out_attn
            
            
            res = x
            x = self.first_resblock[i + 1](x)
            t_emb_out = self.t_emb_layer[i + 1](t_emb)[:,:,None,None]
            x = x + t_emb_out
            x = self.second_resblock[i + 1](x)
            res = self.res_conv_block[i + 1](res)  
            x = x + res
            
        return x
                
            
            
class UpsampleBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, upsample, t_emb_dim,
                 num_layers = 1, num_heads = 4):
        super().__init__()
        self.upsample = upsample
        self.num_layers = num_layers
        
        self.first_resblock = nn.ModuleList(
            [
                nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 
                        kernel_size = 3, padding = 1, stride = 1)
                
                )
                for i in range(num_layers)
            ]
        )
        
        self.t_emb_layer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
            ]
        )
        
        self.second_resblock = nn.ModuleList(
            [
                nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, stride = 1)
                )
                for _ in range(num_layers)   
            
            ]
        )
        
        self.res_conv_block = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels , out_channels, kernel_size = 1)
                for i in range(num_layers)
            ]
        )
        
        self.attention_norm = nn.ModuleList(
            [
                nn.GroupNorm(8, out_channels)
                for _ in range(num_layers)
            ]
        )
        
        self.attention = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim = out_channels, num_heads = num_heads, 
                                      batch_first = True)
                for _ in range(num_layers)
            ]
        )
        
        self.upsample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size = 4, stride = 2, padding = 1) if self.upsample else nn.Identity()
        
    
    def forward(self, x, out_down, t_emb):
        
        x = self.upsample_conv(x)
        x = torch.cat([x, out_down], dim = 1)
        
        for i in range(self.num_layers):
            res = x    
            # First Resblock
            x = self.first_resblock[i](x)
            # Time embedding
            t_emb_out = self.t_emb_layer[i](t_emb)[:,:,None,None]
            x = x + t_emb_out
            # Second Resblock
            x = self.second_resblock[i](x)
            # Residual connection
            res = self.res_conv_block[i](res)
            x = x + res
            
            # Attention
            b, c, h, w = x.shape
            x_reshaped = x.reshape(b, c, h * w)
            in_attn = self.attention_norm[i](x_reshaped)
            in_attn = in_attn.permute(0,2,1)
            out_attn, _ = self.attention[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.permute(0,2,1).reshape(b, c, h, w)
            x = x + out_attn
            
        return x
    
    
class UNetModel(nn.Module):
    
    def __init__(self, model_config):
        super().__init__()
        
        self.in_channels = model_config['in_channels']
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.up_channels = list(reversed(model_config['down_channels']))
        self.t_emb_dim = model_config['time_emb_dim']
        self.num_layers = model_config['num_layers']
        self.num_heads = model_config['num_heads']
        self.down_sample = model_config['down_sample']
        self.up_sample = list(reversed(model_config['down_sample']))
        
        self.conv_in = nn.Conv2d(self.in_channels, self.down_channels[0], kernel_size = 3,
                                 padding = 1, stride = 1)
        
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim * 4, self.t_emb_dim)
        )
        
        self.downs = nn.ModuleList([
            DownsampleBlock(
                in_channels = self.down_channels[i],
                out_channels = self.down_channels[i + 1],
                downsample = self.down_sample[i],
                t_emb_dim = self.t_emb_dim,
                num_layers = self.num_layers,
                num_heads = self.num_heads
            )
            for i in range(len(self.down_channels) - 1)
        ])
        
        self.mid = nn.ModuleList([
            MidBlock(
                in_channels = self.mid_channels[i],
                out_channels = self.mid_channels[i + 1],
                t_emb_dim = self.t_emb_dim,
                num_layers = self.num_layers,
                num_heads = self.num_heads
            )
            for i in range (len(self.mid_channels) - 1)
        ])
        
        self.ups = nn.ModuleList([
            UpsampleBlock(
                in_channels = self.up_channels[i] * 2,
                out_channels = self.up_channels[i + 1] if i != len(self.up_channels) - 2 else 16,
                upsample = self.up_sample[i],
                t_emb_dim = self.t_emb_dim,
                num_layers = self.num_layers,
                num_heads = self.num_heads
            )
            for i in range(len(self.up_channels) - 1)
        ])
        
        
        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, self.in_channels, kernel_size = 3, padding = 1, stride = 1)
        
        
    def forward(self, x, timesteps):
        
        x = self.conv_in(x)
        
        time_emb = get_time_embedding(timesteps, self.t_emb_dim)
        time_emb = self.t_proj(time_emb)
        
        down_outs = []
        
        for downblock in self.downs:
            down_outs.append(x)
            x = downblock(x, time_emb)
            
        
        for midblock in self.mid:
            x = midblock(x, time_emb)
            
        for upblock in self.ups:
            out_down = down_outs.pop()
            x = upblock(x, out_down, time_emb)
            
        x = self.norm_out(x)
        x = nn.SiLU()(x)
        x = self.conv_out(x)
        
        
        return x