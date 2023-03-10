# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:52:53 2023
@author: tomDaSilva
"""

import torch
import torch.nn as nn 
import torchvision.transforms.functional as TF

from torch_blocks import ConvBlock

class UNET(nn.Module) : 
    """ U-Net architecture : Encoder -> Bottleneck -> Decoder 
        @parameter : inCh (first layer input channels) 
        @parameter : outCh (last layer output channels)
        @parameter : features (interm layers nb of channels)
        
        @attribute : name (model name)
        @attribute : upSample (Decoder) 
        @attribute : downSample (Encoder) 
        @attribute : bottleneck 
        @attribute : out (output layer) 
        @attribute : pool (max pooling layer) 
    """

    def __init__(self, name, inCh=3, outCh=1, features=[64, 128, 256, 512]):
        """ Main class constructor """
        
        super(UNET, self).__init__()
        self.name = name
        self.upSample, self.downSample = nn.ModuleList(), nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features : 
            self.downSample.append(ConvBlock(inCh, feature))
            inCh = feature
        
        # Decoder 
        for feature in reversed(features) : 
            self.upSample.append(nn.ConvTranspose2d(feature*2, feature, 
                                                      kernel_size=2, stride=2))
            self.upSample.append(ConvBlock(feature*2, feature))
            
        self.bottleneck = ConvBlock(features[-1], features[-1]*2)
        self.out = nn.Conv2d(features[0], outCh, kernel_size=1)
        
    def forward(self, x) : 
        """ Image propagation """
        
        routeConnections = []
        for down in self.downSample : 
            x = down(x)
            routeConnections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        routeConnections = routeConnections[::-1] # Reversing skip connections
        
        for i in range(0, len(self.upSample), 2) : 
            x = self.upSample[i](x)
            routeConnection = routeConnections[i//2]
            if x.shape != routeConnection.shape:
                x = TF.resize(x, size=routeConnection.shape[2:])
                
            concatRoute = torch.cat((routeConnection, x), dim=1)
            x = self.upSample[i+1](concatRoute)
            
        return self.out(x)
    
    