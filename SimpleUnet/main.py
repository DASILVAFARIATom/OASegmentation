# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:53:29 2023
@author: tomDaSilva
"""

#%% imports
import os
import numpy as np
from datetime import datetime

from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader

from torch_metrics import Loss
from torch_dataset import AODataset, generate_paths
from torch_unet import UNET
from torch_training import train_loop

#%% variables
IMG_HEIGHT, IMG_WIDTH = 512, 512
SPLIT_RATE = 95 # Amount of images in train dataset
BATCH_SIZE = 8  # Amount of images used in one training batch
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = "cuda:0"
CKP_STEP = 5
if os.path.exists("/projet1/tdasilva/Dataset/Patients"):
    DATASET_PATH = "/projet1/tdasilva/Dataset/Patients"
else: 
    DATASET_PATH = "../Dataset/Patients"

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
    
    
    #%% model + parametre
    # Creating model, optimizer and loss function
    unet = UNET("simpleUnet", inCh=1, outCh=1).to(DEVICE)
    loss = Loss(name = "FTLoss")
    opt  = Adam(unet.parameters(), lr=LEARNING_RATE)
    
    #%% trainning
    # Train steps and loss dict
    tStep, sStep = len(trainDataset)/BATCH_SIZE, len(valDataset)/BATCH_SIZE
    if tStep > int(tStep) : tStep = int(tStep) + 1
    if sStep > int(sStep) : sStep = int(sStep) + 1
    tStep, sStep = int(tStep), int(sStep)
    
    H, trainedUnet = train_loop(unet, trainLoader, valLoader, loss, opt,
                                EPOCHS, tStep, sStep, DEVICE, CKP_STEP)
    
    now = datetime.now()   
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    filename = f"Checkpoints/{unet.name}_{loss.name}_{dt_string}.npz"
    np.savez(filename, **H)
    
    