# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:52:53 2023
@author: tomDaSilva
"""

import torch.nn as nn 

class ConvBlock(nn.Module) : 
    """ Convolutional block : Conv2D->BatchNorm->ReLU x2
        @parameter : inCh (amount of input channels)
        @parameter : outCh (amount of output channels)
        @parameter : kSize (conv kernel size)
        @parameter : stride (kernel moving step)
        @parameter : padding ('valid' or 'same')
    
        @attribute : conv (set of nn layers -> conv block)
    """
    
    def __init__(self, inCh, outCh, nConv=2, kSize=3, stride=1, padding=1) : 
        """ Main class constructor """
        
        super(ConvBlock, self).__init__() 
        
        self.conv = nn.Sequential() 
        for cpt in range(nConv) : 
            self.conv.add_module(f"conv{cpt}", nn.Conv2d(inCh, outCh, kSize, stride, padding, bias=False))
            self.conv.add_module(f"batchNorm{cpt}", nn.BatchNorm2d(outCh))
            self.conv.add_module(f"ReLU{cpt}", nn.ReLU(inplace=True))
            inCh = outCh
     
    def forward(self, x) : 
        """ Image propagation """
        return self.conv(x)