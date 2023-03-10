# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:37:12 2023
@author: tomDaSilva
"""

import torch.nn as nn 
from torch import cat
from torch_blocks import StemBlock, InceptionResNetA, ReductionA, FireSqueeze
from torch_blocks import InceptionResNetB, ReductionB, InceptionResNetC, DoubleConvBlock

from CONFIG import *

class InceptionResnetV2(nn.Module) : 
    """ InceptionResnetV2 Archtitecture 
    
        @parameter : inCh (first layer input channels) 
        @parameter : fCh  (first conv output channels)
        
        @attribute : upSample (Decoder) 
        @attribute : downSample (Encoder) 
        @attribute : bottleneck 
        @attribute : name (model name)
    """
    
    def __init__(self, name, inCh=1, outCh=1, nBlocks=5) :
        """ Main class constructor """
        
        super(InceptionResnetV2, self).__init__()
        self.name, self.nBlocks = name, nBlocks
        self.upSample, self.downSample = nn.ModuleList(), nn.ModuleList()
        
        # Encoder 
        features = FEATURES
        self.downSample.append(StemBlock(inCh, outCh=features[0]))
        for _ in range(nBlocks) :
            self.downSample.append(InceptionResNetA(features[0]))
        self.downSample.append(ReductionA(features[0], features[1]))
        for _ in range(2*nBlocks) : 
            self.downSample.append(InceptionResNetB(features[1]))
        self.downSample.append(ReductionB(features[1], features[2]))
        for _ in range(nBlocks) : 
            self.downSample.append(InceptionResNetC(features[2]))
        self.downSample.append(nn.AdaptiveAvgPool2d((IMG_HEIGHT//16, IMG_WIDTH//16)))
        self.downSample.append(nn.Dropout2d(1 - 0.8))
            
        # Bottleneck 
        self.bottleneck = FireSqueeze(features[2])
        self.preSample = nn.ConvTranspose2d(features[2]//2, features[2]//2, kernel_size=1, stride=1)
            
        # Decoder
        self.upSample.append(nn.Sequential(
            nn.ConvTranspose2d(features[2] + features[2]//2, features[2], kernel_size=2, stride=2), 
            DoubleConvBlock(features[2], features[2], kSize=3, padding=1)))
        self.upSample.append(nn.Sequential(
            nn.ConvTranspose2d(features[2]*2, features[2], kernel_size=2, stride=2), 
            DoubleConvBlock(features[2], features[1], kSize=3, padding=1)))
        self.upSample.append(nn.Sequential(
            nn.ConvTranspose2d(features[1]*2, features[1], kernel_size=2, stride=2), 
            DoubleConvBlock(features[1], features[0], kSize=3, padding=1)))
        self.upSample.append(nn.Sequential(
            nn.ConvTranspose2d(features[0], features[0]//2, kernel_size=2, stride=2), 
            DoubleConvBlock(features[0]//2, features[0]//4, kSize=3, padding=1)))
        
        self.out = nn.Conv2d(features[0]//4, outCh, kernel_size=1)
        
    def forward(self, x) :
        
        routeConnections = []
        for cpt, down in enumerate(self.downSample) :
            x = down(x)
            if((cpt == 16) or (cpt == 22) or (cpt == 23) ) :
                routeConnections.append(x)
        
        routeConnections = routeConnections[::-1]
        x = self.bottleneck(x)
        x = self.preSample(x)
        
        for cpt, up in enumerate(self.upSample) : 
            if(cpt <= 2) : x = cat((routeConnections[cpt], x), 1)
            x = up(x)
            
        return self.out(x)
