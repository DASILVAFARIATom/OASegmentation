# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:45:12 2023
@author: tomDaSilva
"""

from torch import save

import matplotlib.pyplot as plt
from IPython.display import clear_output

def save_checkpoints(state, filename) :
    print(f"[INFO] - Saving checkpoint in :\n{filename}\n")
    save(state, filename)
    
def format_seconds_to_hhmmss(seconds):
    hours = seconds // (60*60)
    seconds %= (60*60)
    minutes = seconds // 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds) 

def live_plot(data_dict, image, size=(7,5)):
    """ Sequentially updating a plot : results and segmentation """
    clear_output(wait=True)
    fig = plt.figure(figsize=size)
    for i in range(2) : 
        ax = fig.add_subplot(1, 2, i+1)
        if i == 0 : 
            for label,data in data_dict.items(): ax.plot(data, label=label)
            plt.title("Training results")
            plt.grid(True)
            plt.xlabel('epoch')
            plt.legend(loc='center left') # the plot evolves to the right
        else : 
            ax.imshow(image, cmap="Greys_r")
            plt.title("Predicted segmentation")
    plt.show()
    
def PadifNeeded(img, label, n_dim=512):
    old_shape = img.shape
    if old_shape[1] != n_dim and old_shape[2] != n_dim:
        min_value = img.min()
        img[img == 0] = min_value
        Pad_coord = [int((n_dim/2) - (old_shape[2]/2)), int((n_dim/2) - (old_shape[3]/2)),old_shape[2],old_shape[3] ]
        img[:,Pad_coord[0]:Pad_coord[0]+Pad_coord[2],Pad_coord[1]:Pad_coord[1]+Pad_coord[3]]  = img
        label[:,Pad_coord[0]:Pad_coord[0]+Pad_coord[2],Pad_coord[1]:Pad_coord[1]+Pad_coord[3]]  = label
    else:
        Pad_coord = [0,0,old_shape[1],old_shape[2]]
 
    return img,label, Pad_coord
