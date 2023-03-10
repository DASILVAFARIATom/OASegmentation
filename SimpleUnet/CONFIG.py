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
LEARNING_RATE = 0.1 
EPOCHS = 100

# Config parameters 
DEVICE = "cuda:0"
CKP_STEP = 10
if os.path.exists("/projet1/tdasilva/Dataset/Patients"):
    DATASET_PATH = "/projet1/tdasilva/Dataset/Patients"
else : DATASET_PATH = "../Dataset/Patients"

# Checkpoints parameters
SAVE_PATH = "./Checkpoints/InceptionResnetV2_Tversky_epoch10_09032023_150853.pth"
RSLT_PATH = "./Checkpoints/InceptionResnetV2_Tversky_09032023_150853.npz"