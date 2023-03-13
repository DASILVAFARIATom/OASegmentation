# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:05:56 2023
@author: tomDaSilva
"""

import torch
import torch.nn as nn 

###########################
ALPHA_F = 0.8
GAMMA_F = 2
ALPHA_T = 0.5
BETA_T  = 0.5

ALPHA_TF = 0.5
BETA_TF  = 0.5
GAMMA_TF = 1
###########################

def computeIoULoss(x, y, smooth=1) :
    """ Computing IoU loss (Jaccard index) 
        @parameter : x (prediction)
        @parameter : y (GT)
    """
    intersection = (x * y).sum()
    total = (x + y).sum()
    union = total - intersection 
    IoU = (intersection + smooth)/(union + smooth)
    return 1 - IoU

def computeDiceLoss(x, y, smooth=1) :
    """ Computing Dice loss """
    intersection = (x * y).sum()                            
    dice = (2.*intersection + smooth)/(x.sum() + y.sum() + smooth)  
    return 1 - dice

def computeFocalLoss(x, y, smooth=1, alpha=0.8, gamma=2) : 
    """ Computing Focal loss"""
    BCE = torch.binary_cross_entropy(x, y, reduction='mean')
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
    return focal_loss

def computeTverskyLoss(x, y, smooth=1, alpha=0.5, beta=0.5) : 
    """ Computinf Tversky loss """
    TP = (x * y).sum()    
    FP = ((1-y) * x).sum()
    FN = (y * (1-x)).sum()
   
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    return 1 - Tversky

def computeFTLoss(x, y, smooth=1, alpha=0.5, beta=0.5, gamma=1) :
    """ Computing Focal-Tversky loss """
    TP = (x * y).sum()    
    FP = ((1-y) * x).sum()
    FN = (y * (1-x)).sum()
    
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    FocalTversky = (1 - Tversky)**gamma
                   
    return FocalTversky

def arctan_activation(x) : 
    """ Computing arctangent activation function """
    pi = torch.acos(torch.zeros(1)).item()*2
    return 1e-7 + (1 - 2*1e-7)*(0.5 + torch.arctan(x)/torch.tensor(pi))

class Loss(nn.Module) : 
    
    def __init__(self, name="IoULoss", weights=None, size_average=True) :
        super(Loss, self).__init__()
        self.name = name
        if self.name=="IoULoss"     : print("Chosen loss : Jaccard")
        elif self.name=="DiceLoss"  : print("Chosen loss : Dice loss")
        elif self.name=="FocalLoss" : print("Chosen loss : Focal loss")
        elif self.name=="FTLoss"    : print("Chosen loss : Focal Tversky loss")
        else : print("Chosen loss : Tversky loss")
        
    def forward(self, inputs, targets) :
        inputs = arctan_activation(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        if self.name=="IoULoss"     : return computeIoULoss(inputs, targets)
        elif self.name=="DiceLoss"  : return computeDiceLoss(inputs, targets)
        elif self.name=="FocalLoss" : return computeFocalLoss(inputs, targets)
        elif self.name=="FTLoss"    : return computeFTLoss(inputs, targets)  
        else : return computeTverskyLoss(inputs, targets)