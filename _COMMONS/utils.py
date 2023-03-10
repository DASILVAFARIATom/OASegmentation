# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:45:12 2023
@author: tomDaSilva
"""

from torch import save

import matplotlib.pyplot as plt
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