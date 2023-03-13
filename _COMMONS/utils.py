# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:45:12 2023
@author: tomDaSilva
"""

import numpy as np 
import matplotlib.pyplot as plt

from torch import save
from IPython.display import clear_output

def save_checkpoints(state, filename) :
    print("[INFO] - Saving checkpoint")
    save(state, filename)
    
def format_seconds_to_hhmmss(seconds):
    hours = seconds // (60*60)
    seconds %= (60*60)
    minutes = seconds // 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds) 

def split_image(im, size):
    
    im_width, im_height = im.shape
    row_width, row_height = size
    cols, rows = int(im_width/row_width), int(im_height/row_height)
    
    output = np.zeros(( ((im.shape[0]*im.shape[1])//(row_width*row_height)), row_width, row_height ))
    
    n = 0
    for j in range(0, rows) :
        for i in range(0, cols) :    
            box = (j * row_width, i * row_height, j * row_width + row_width, i * row_height + row_height)
            output[n] = im[box[0]:box[2], box[1]:box[3]]            
            n += 1
    
    return output
    

def live_plot(data_dict, image, segm, gt, size=(10,3), live=True):
    """ Sequentially updating a plot : results and segmentation """
    if live : clear_output(wait=True)
    fig = plt.figure(figsize=size)
    for i in range(4) : 
        ax = fig.add_subplot(1, 4, i+1)
        if i == 0 : 
            for label,data in data_dict.items(): ax.plot(data, label=label)
            plt.title("Training results")
            plt.grid(True)
            plt.xlabel('epoch')
            plt.legend(loc='center left') # the plot evolves to the right
        elif i == 1 :
            ax.imshow(image, cmap="Greys_r")
            plt.title("Original image")
        elif i == 2 : 
            ax.imshow(gt, cmap="Greys_r")
            plt.title("Ground Truth")
        else : 
            ax.imshow(segm, cmap="Greys_r")
            plt.title("Prediction")
            
    plt.show()

def plot_images(images, titles, n, size) : 
    """ Plotting a raw of images """
    fig = plt.figure(n, figsize=size)
    for i in range(len(images)) : 
        ax = fig.add_subplot(1, len(images), i+1)
        ax.imshow(images[i], cmap="Greys_r")
        plt.title(titles[i])
    plt.show()
    