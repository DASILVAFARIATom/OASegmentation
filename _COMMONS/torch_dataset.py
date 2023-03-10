# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 09:55:03 2023
@author: tomDaSilva
"""

import os
from torch import max
import numpy as np

from torch.utils.data import Dataset

def generate_paths(path, r=80) :
    """ Retrieving dataset from main patients path 
        @parameter : path (path where all the patients folder are stored)
        @parameter : r (amount of images inside train set)
        
        .npz files structure -> "x" : images -- "y" : GT
    """
    
    # Subsets : Train, test and validation
    trainSet, valSet, testSet = [], [], []
    
    # Getting names of all the patients (folders names)
    patients = [name for name in os.listdir(path) 
                  if os.path.isdir(os.path.join(path, name))]
    nPatients = len(patients) 
    filesList = sorted([os.path.join(root, f) # All of the .npz files
                        for root, _, files in os.walk(path)
                            for f in files if (f.endswith(".npz"))])
    tR, tS = int(nPatients*r/100), int((nPatients*(100-r)/100)/2)
    
    patientsTrain = patients[0:tR]
    patientsValid = patients[tR:tR+tS]
    
    for f in filesList : 
        try : patient = f.split('\\')[1]
        except : patient = f.split('/')[5]
        if patient in patientsTrain : trainSet.append(f)
        elif patient in patientsValid : valSet.append(f)
        else : testSet.append(f)
        
    return trainSet, valSet, testSet

class AODataset(Dataset) :
    """ Torch Dataset
        @parameter : paths (paths to subset images)
        
        @attribute : files (.npz files path)
        @attribute : transf (data augmentation)
    """
    
    def __init__(self, paths, transf=None):
        """ Main class constructor """
        self.transf, self.files = transf, paths
        print("Loaded dataset : {} images".format(len(self.files)))
        
    def __len__(self) : 
        """ Returning amount of annotated images in the whole dataset """
        return len(self.files)
    
    def __getitem__(self, idx) : 
        """ Returning image [idx], its GT and the patient name """
        
        filepath = self.files[idx]
        with np.load(filepath) as data : 
            x, y = data["x"].astype(np.float32), data["y"].astype(np.uint8)
        x, y = x.transpose((1, 2, 0)), y.transpose((1, 2, 0))
        if self.transf is not None : x, y = self.transf(x), self.transf(y)
        y = y / max(y)
        return (x, y, self.files[idx])