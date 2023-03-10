# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:53:29 2023
@author: tomDaSilva
"""

#%% imports
import sys
sys.path.append("./_MODEL")
sys.path.append("./_UTILS")

import numpy as np
from datetime import datetime

from torch.optim import Adam, lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader

from torch_metrics import Loss
from torch_dataset import AODataset, generate_paths
from torch_unet import UNET
from torch_training import train_loop

from CONFIG import *

#%% dataloader
if __name__ == "__main__":
    
    train, val, test = generate_paths(DATASET_PATH, SPLIT_RATE)
    
    # Images transformations/augmentation
    transf = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)), 
            transforms.ToTensor()
        ])
    
    # Loading dataset (train, test and validation)
    trainDataset = AODataset(train, transf=transf)
    valDataset   = AODataset(val,   transf=transf)
    
    # Creating dataloaders
    trainLoader = DataLoader(trainDataset, shuffle=True, batch_size=BATCH_SIZE,
                             pin_memory=False)#, num_workers=os.cpu_count())
    valLoader   = DataLoader(valDataset, shuffle=False, batch_size=BATCH_SIZE, 
                             pin_memory=False)#, num_workers=os.cpu_count())
    
    
    #%% model + params
    unet = UNET("simpleUnet", inCh=1, outCh=1).to(DEVICE)
    loss = Loss(name = "FTLoss")
    opt  = Adam(unet.parameters(), lr=LEARNING_RATE)
    sch  = lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, 
                                          patience=2, min_lr=0.0001)
    
    #%% Training    
    tStep, sStep = len(trainDataset)/BATCH_SIZE, len(valDataset)/BATCH_SIZE
    if tStep > int(tStep) : tStep = int(tStep) + 1
    if sStep > int(sStep) : sStep = int(sStep) + 1
    tStep, sStep = int(tStep), int(sStep)
    
    # launching train loop
    H, trainedUnet = train_loop(unet, trainLoader, valLoader, loss, opt, sch,
                                EPOCHS, tStep, sStep, DEVICE, CKP_STEP)
    
    now = datetime.now()   
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    filename = f"Checkpoints/{unet.name}_{loss.name}_{dt_string}.npz"
    np.savez(filename, **H)
    
    