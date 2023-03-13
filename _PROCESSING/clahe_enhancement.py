# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:30:56 2023
@author: tomDaSilva
"""

import numpy as np
import scipy

from tqdm import tqdm

def compute_kernel(size, thetaInc, scales) : 
    """ Computing rotating kernels 
    
        @parameter size (kernel size)
        @parameter thetaInc (angles step)
        @parameter scales (kernel scales -> amount of considered scales)
    """    
    
    M, T  = size 
    theta = (np.arange(0, 180, thetaInc)) * (np.pi / 180)
    w_d   = np.linspace(5, 9, scales)
    w_c   = ((M+1)/2) - 0.75*w_d
    
    print("w_c =", w_c)
    
    alpha = np.log10(np.cosh((M-1) * np.arccosh((1/np.cos(np.pi * w_c/M)))))
    gamma, beta = 1/(M+1), np.cosh((1/M)*np.arccosh(10**alpha))
    
    print("alpha = {} -- beta = {}".format(alpha, beta))
    print("gamma =", gamma)
    kernel = np.zeros((size[0], size[1], len(theta), len(w_d)))
    
    for idx_w, w_cj in enumerate(w_c) : 
        for idx_t, t in enumerate(theta) : 
            mat_mul = np.array([[np.cos(t), np.sin(t)], 
                                [-np.sin(t), np.cos(t)]])
            xy = np.array(np.meshgrid(np.arange(0, kernel.shape[0]), 
                                      np.arange(0, kernel.shape[1]), 
                                      indexing='ij'))
            xy = np.array([xy[0].flatten(), xy[1].flatten()]).T
            uv = xy @ mat_mul
            
            A = beta[idx_w] * np.cos(np.pi * uv[:, 0] * gamma)
            idx_a, idx_b = xy[np.where(np.abs(A)<=1)], xy[np.where(np.abs(A)>1)]
            
            kernel[idx_a[:, 0], idx_a[:, 1], idx_t, idx_w] = np.abs(
                np.cos(M*np.arccos(beta[idx_w]*
                         np.cos(np.pi*uv[np.where(np.abs(A) <= 1)][:, 0]*
                         gamma))
                )
            )
            kernel[idx_b[:, 0], idx_b[:, 1], idx_t, idx_w] = np.abs(
                np.cosh(M*np.arccosh(beta[idx_w]))
            )
            
            m_ij = np.mean(kernel[:, :, idx_t, idx_w])
            kernel[:, :, idx_t, idx_w] -= m_ij

    return kernel

def compute_convolution(imageCLAHE, kernel) : 
    """ Computing convolutions for each kernels 
    
        @parameter imageCLAHE (enhanced histogram CLAHE image)
        @parameter kernel (rotated and scaled kernels)
    """
    
    output_images = np.zeros((imageCLAHE.shape[0], imageCLAHE.shape[1], 
                              kernel.shape[2], kernel.shape[3]))
    
    for idx_w in range(kernel.shape[3]) :
        for idx_t in tqdm(range(kernel.shape[2])) : 
            output_images[:, :, idx_t, idx_w] = scipy.ndimage.convolve(
                imageCLAHE, kernel[:, :, idx_t, idx_w]
            )
            
    max_image = np.max(output_images, axis=2)
    f_b = (1/kernel.shape[3])*np.sum(max_image, axis=-1)
    f_b = f_b/np.max(f_b)
    
    return output_images, max_image, f_b