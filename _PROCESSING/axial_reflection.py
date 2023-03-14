# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 10:04:12 2023
@author: tomDaSilva
"""

import numpy as np
import skimage, scipy, sys, os, sklearn

sys.path.append("../_COMMONS")
from tqdm import tqdm 

from sklearn.cluster import KMeans
from clahe_enhancement import compute_convolution, compute_kernel
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
    """ Returns the <n> largest connected elements on <binaryImg> 
        
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
    Sm = ( (1/(np.shape(np.where(IEs))[1]))*sum_it2 )
    
    # Computing binary mask
    Im = np.zeros(np.shape(It2))
    Im[np.where(It2 >= Sm)] = 1
    
    return np.bitwise_or(Im.astype(np.uint8), IEs.astype(np.uint8))
    
def dilation_reconstruction(It2, Im, radius=1, nIter=10000) :
    """ Geodesic dilation of It2 
    
        @parameter It2 (orientations image)
        @parameter Im (geodesic mask)
    """
    
    k = np.ones((radius*2 + 1, )*2, dtype=np.uint8)
    print(k)
    marker = It2
    for _ in tqdm(range(nIter)) : 
        expanded = skimage.morphology.dilation(marker, k)
        expanded = np.minimum(expanded, Im)
        
        if (marker == expanded).all():
            return expanded
        
        marker = expanded
    return expanded

def kmeans_clustering(img, nClusters) : 
    """ Performing image quantification with kmeans algorithm 
    
        @parameter img (image to be processed)
        @parameter nClusters (amount of targeted clusters)
    """
    
    X   = img.reshape((-1, 1))
    kM  = KMeans(n_clusters=nClusters, n_init=4)
    kM.fit(X)
    
    values = kM.cluster_centers_.squeeze()
    labels = kM.labels_
    
    return values, labels
    
if __name__ == "__main__" :
    
    
    filepath = DATASET_PATH+IMG_2
    with np.load(filepath) as data : 
        x, y = data["x"].astype(np.float32)[0], data["y"].astype(np.uint8)[0]
    
    #%% 1 - Pre-Processing
    
    # Image smooting
    clahe = skimage.exposure.equalize_adapthist(x, clip_limit=0.1)
    filtered = skimage.filters.median(clahe, skimage.morphology.square(1))
    closed   = skimage.morphology.closing(filtered, skimage.morphology.disk(1))
    topHat   = skimage.morphology.white_tophat(closed, skimage.morphology.disk(11))
    
    #%% 2 - Detection of bright elongated structures 
    meanGreys = means_greys(closed, segmLen=55)
    orientations = direction_estimation(meanGreys, thresh=0.95)
    reflexions   = get_axial_reflections(topHat, orientations, segmLen=15)
    
    low, high = 0.2, 0.6
    lowt = (topHat > low).astype(int)
    hight = (topHat > high).astype(int)
    hyst = skimage.filters.apply_hysteresis_threshold(topHat, low, high)
    hyst = get_largest_connected(hyst, n=10)
    
    
    mask = compute_geodesic_mask(reflexions, hyst)
    skeleton = dilation_reconstruction(reflexions, mask)
    skeleton = skeleton > 0.5
    skeleton = skimage.morphology.erosion(skeleton, skimage.morphology.square(7))
    skeleton = get_largest_connected(skeleton, n=5)
    
    #%% 3 - Classification of darkest areas
    values, labels = kmeans_clustering(x, nClusters=3)
    clustered = np.choose(labels, values)
    clustered.shape = x.shape
    
    plot_images([clustered], ["kmeans"], 1, (9, 9))
          
    
    # skeleton = np.bitwise_and(skeleton.astype(np.uint8), connected.astype(np.uint8))
    # plot_images([x, clahe, topHat, reflexions, hyst, skeleton, clustered],
    #             ["image", "CLAHE", "tophat", "reflexions", "Thresh", "skeleton", "kmeans"], 
    #             1, (9, 9))
      
    
    
    
