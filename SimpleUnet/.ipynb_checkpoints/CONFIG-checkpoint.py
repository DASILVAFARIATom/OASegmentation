# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:55:40 2023
@author: tomDaSilva
"""

import os 

# Images parameters
IMG_HEIGHT, IMG_WIDTH = 512, 512

# Learning parameters
SPLIT_RATE = 95 # Amount of images in train dataset
BATCH_SIZE = 16 # Amount of images used in one training batch
LEARNING_RATE = 0.001 
EPOCHS = 20
LOSS = "Tversky"

# Config parameters 
DEVICE = "cuda:0"
CKP_STEP = 10
if os.path.exists("/projet1/tdasilva/Dataset/Patients"):
    DATASET_PATH = "/projet1/tdasilva/Dataset/Patients"
else : DATASET_PATH = "../../ML/Dataset/Patients"