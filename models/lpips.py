from __future__ import absolute_import
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import torch.nn
import torchvision




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def spatial_average(tensor, keepdim=True):
    return tensor.mean([2, 3], keepdim=keepdim)



class vgg16(nn.Module):
    
    def __init__(self, requires_grad=False, pretrained = True):
        
        
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=pretrained).features
        
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()   
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
            
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
            
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
        
    def forward(self, X):
        
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out



class LPIPS(nn.Module):
    
    def __init__(self, net='vgg', version='0.1', dropout = True):
        super(LPIPS, self).__init__()
        
        # For imagenet
        self.scaling_layer = ScalingLayer()
        
        
        if net == 'vgg':
            self.net = vgg16().to(device)
            self.chns = [64, 128, 256, 512, 512]
            self.lenchains = len(self.chns)
        else:
            raise NotImplementedError('Only VGG16 is implemented')
        
        
        
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=dropout).to(device)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=dropout).to(device)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=dropout).to(device)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=dropout).to(device)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=dropout).to(device)
        self.lins = nn.ModuleList([self.lin0, self.lin1, self.lin2, self.lin3, self.lin4])
        
        
        # Load the weights of trained LPIPS model
        import inspect
        import os
        model_path = os.path.abspath(
            os.path.join(inspect.getfile(self.__init__), '..', 'weights/v%s/%s.pth' % (version, net)))
        print('Loading model from: %s' % model_path)
        self.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        ########################
        
        # Freeze all parameters
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        ########################
        
        
    def forward(self,in0, in1, normalize = False):
        
        if normalize:
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1
            
        in0_input = self.scaling_layer(in0)
        in1_input = self.scaling_layer(in1)
        
        outs0 = self.net(in0_input)
        outs1 = self.net(in1_input)
        
        feats0 , feats1, diffs = {}, {}, {}
        
        
        for kk in range(self.lenchains):
            
            feats0[kk] = torch.nn.functional.normalize(outs0[kk], dim=1)
            feats1[kk] = torch.nn.functional.normalize(outs1[kk], dim=1)
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
            
        
        res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.lenchains)]
        
        val = 0
        
        for k in range(self.lenchains):
            val += res[k]
        return val    
    


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        # Imagnet normalization for (0-1)
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])
    
    def forward(self, inp):
        return (inp - self.shift) / self.scale
    

class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.model(x)
        return out