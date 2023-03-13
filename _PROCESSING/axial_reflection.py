# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 10:04:12 2023
@author: tomDaSilva
"""

import numpy as np
import skimage, scipy, sys, os

sys.path.append("../_COMMONS")
from tqdm import tqdm 
from utils import plot_images

if os.path.exists("/projet1/tdasilva/Dataset/Patients"):
    DATASET_PATH = "/projet1/tdasilva/Dataset/Patients"
else : DATASET_PATH = "../../ML/Dataset/Patients"

IMG_1 = "/ADAT245/OD_20141223114356_X0.0T_Y0.0_Z250.0.npz"
IMG_2 = "/ADAT245/OD_20141223114455_X0.0T_Y0.0_Z260.0.npz"
IMG_3 = "/ADAT245/OD_20141223114533_X0.0T_Y0.0_Z260.0.npz"
IMG_4 = "/ADAT245/OD_20141223114604_X0.0T_Y0.0_Z260.0.npz"

def means_greys(image, segmLen=55, N=5) :
    """ Computing means of the image on a <segmLen> long segment 
        at different orientations
    
        @parameter image (input image) 
        @parameter segm_len (segment len (segment on which mean is computed))
        @parameter N (orientation incrementation step) 
    """
    
    angles = np.arange(0, 180, N)
    mean_mat = np.zeros((image.shape[0], image.shape[1], len(angles)))
    
    for cpt, a in enumerate(tqdm(angles)) : 
        elem = np.zeros((segmLen, segmLen))
        elem[int(segmLen/2), :] = 1
        elem = skimage.transform.rotate(elem, a, resize=False)
        elem[np.where(elem > 0)] = 1
        elements = np.count_nonzero(elem)
        
        mean_mat[:, :, cpt] = scipy.ndimage.convolve(image, elem)/elements
        
    return mean_mat

def direction_estimation(means, thresh=0.75) :
    """ Computing local shape direction based on the previously computed 
        oriented means
        
        @parameter means (oriented means)
    """    
    
    N = means.shape[-1]
    k = np.argmax(means, axis=-1)
    
    Id = -1 * np.ones((k.shape[0], k.shape[1]))
    for i in tqdm(range(k.shape[0])) : 
        for j in range(k.shape[1]) : 
            if (thresh * means[i, j, k[i, j]] > means[i, j, int((k[i, j] + (N/2)) % N)]) :
                    Id[i, j] = k[i, j]
                    
    return Id

def get_axial_reflections(It1, Id, segmLen=21, N=5) :
    """ Computing vessels axial reflections from top-hat image and local 
        directions 
        
        @parameter It1 (Top-Hat image)
        @parameter Id (local directions (computed with <direction_estimation>))
        @parameter segmLen (Morpholigical closing operator len)
        @parameter N (angles step)
    """
    
    angles = np.arange(0, 180, N)
    It2_mat, It2 = np.zeros((Id.shape[0], Id.shape[1], len(angles))), np.zeros((Id.shape[0], Id.shape[1]))
    
    for cpt, a in enumerate(tqdm(angles)) :
        elem = np.zeros((segmLen, segmLen))
        elem[int(segmLen/2), :] = 1
        elem = skimage.transform.rotate(elem, a, resize=False)
        elem[np.where(elem > 0)] = 1

        It2_mat[:, :, cpt] = skimage.morphology.closing(It1, elem)
        
    for i in range(It2.shape[0]) :
        for j in range(It2.shape[1]) : 
            if(Id[i, j] >= 0) :
                It2[i, j] = It2_mat[i, j, int(Id[i, j])]
            else : 
                It2[i, j] = It1[i, j]
    return It2

def get_largest_connected(binaryImg, n=5) :
    """ Returning the <n> largest connected elements on <binaryImg> 
        
        @parameter binaryImg (binary image to be processed)
        @parameter n (number of connected components that need to be returned)
    """
    
    labels = skimage.measure.label(binaryImg)
    
    out = np.zeros((n, binaryImg.shape[0], binaryImg.shape[1]))
    for i in range(n) : 
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        out[i] = largestCC
        labels[largestCC] = 0

    return np.sum(out, axis=0)

def compute_geodesic_mask(It2, IEs) : 
    """ Computing binary mask for goedesic dilation 
    
        @parameter It2 (axial reflexions enhanced image)
        @parameter IEs (n largest connected components)
    """
    
    # Computing Sm threshold 
    sum_it2 = 0
    for i in range(It2.shape[0]) : 
        for j in range(It2.shape[1]) : 
            if(IEs[i, j] == 1) : sum_it2 += It2[i, j]
    Sm = ( (1/(IEs.shape[0]*IEs.shape[1]))*sum_it2 ) 
    print("Threshold :", Sm)
    
    
    
if __name__ == "__main__" :
    
    # Loading image
    filepath = DATASET_PATH+IMG_1
    with np.load(filepath) as data : 
        x, y = data["x"].astype(np.float32)[0], data["y"].astype(np.uint8)[0]
    
    # 1 - Pre-Processing
    filtered = skimage.filters.median(x, skimage.morphology.square(7))
    closed   = skimage.morphology.closing(filtered, skimage.morphology.disk(7))
    
    # 2 - Enhancing bright elongated structures 
    topHat = skimage.morphology.white_tophat(closed, skimage.morphology.disk(10))
    meanGreys = means_greys(topHat, segmLen=15)
    orientations = direction_estimation(meanGreys, thresh=0.95)
    reflexions   = get_axial_reflections(topHat, orientations, segmLen=15)
    
    # Detection of bright elongated structures
    low, high = 0.1, 0.2
    lowt = (topHat > low).astype(int)
    hight = (topHat > high).astype(int)
    hyst = skimage.filters.apply_hysteresis_threshold(topHat, low, high)
    
    connected = get_largest_connected(hyst)
    mask = compute_geodesic_mask(reflexions, connected)
    
    
    plot_images([x, hyst, connected], 
                ["Original", "Threshold", "Largest"], 
                1, (9, 9))
    
    
