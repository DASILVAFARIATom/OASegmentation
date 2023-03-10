# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:15:45 2023
@author: tomDaSilva
"""

import torch, os, sys
import numpy as np 
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm 
sys.path.append("./_MODEL")
sys.path.append("./_UTILS")

from torch_dataset import AODataset, generate_paths
from torch_unet import UNET
from utils import live_plot

from CONFIG import *


def test_model(model, tLoad, rsltPath, dev="cuda:0") :
    """ Testing model : plotting training results + test segmentation
        @parameter : model (torch segmentation model)
        @parameter : tLoad (test data loader)
        @parameter : dev (chosen device : default GPU 0)
    """
    
    H = np.load(rsltPath)
    img = torch.zeros(IMG_WIDTH, IMG_HEIGHT, 1)
    live_plot(H, img, img, img, live=False)
    
    with torch.no_grad() :
        model.eval()
        
        tqdm._instances.clear() 
        testLoop = tqdm(tLoad, position=0, leave=True)
        
        for (_, (x, y, _)) in enumerate(testLoop):
            x = x.to(dev)
            pred = model(x)
            
            img  = x.permute(0, 2, 3, 1)[0]
            segm = pred.permute(0, 2, 3, 1)[0]
            gt   = y.permute(0, 2, 3, 1)[0]
            live_plot(H, img.cpu().detach().numpy(), 
                      segm.cpu().detach().numpy(), 
                      gt.cpu().detach().numpy(), live=False)