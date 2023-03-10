# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:24:37 2023
@author: tomDaSilva
"""

import time
from datetime import datetime
import torch
from torchmetrics.functional import dice
from tqdm import tqdm 
import gc
from utils import save_checkpoints, format_seconds_to_hhmmss, live_plot

from CONFIG import *

def train_loop(model, tLoad, vLoad, lossF, opt, sch, 
               nEp, tStep, sStep, dev="cuda", ckpStep=3, plot=False): 
    """ Model training function 
        @parameter : model (torch model)
        @parameter : tLoad (train DataLoader)
        @parameter : vLoad (validation DataLoader)
        @parameter : lossF (loss function)
        @parameter : opt (optimizer)
        @parameter : sch (scheduler - reducing learning rate on plateau)
        @parameter : nEp (amount of epochs for model training)
        @parameter : tStep (training step - amount of batches seen in 1 ep.)
        @parameter : sStep (validation step)
        @parameter : dev (chosen device - "cuda" = GPU)
        @parameter : ckpStep (model weights saving every <ckpStep> ep.)
        @parameter : plot (live plotting validation results)
        
        Returns H (losses and metrics), trained model and saved weights path
    """
    
    startTime = time.time()
    H = {"train_loss":[], "valid_loss":[], "train_dice":[], "valid_dice":[]}
    if plot : 
        img = torch.zeros(IMG_WIDTH, IMG_HEIGHT, 1)
        live_plot(H, img, img, img)
        
    print("\n[INFO] - Launching model training :")
    gc.collect()
    torch.cuda.empty_cache()
    for e in range(nEp) : # Iterating over epochs 
        model.train() # Training mode
        totalTrainLoss, totalTestLoss = 0, 0
        totalTrainDice, totalTestDice = 0, 0
        
        # Training progress bar set up 
        tqdm._instances.clear() 
        trainLoop = tqdm(tLoad, position=0, leave=True)
        
        for (_, (x, y, n)) in enumerate(trainLoop) : # Iterating over 1 batch
            x = x.to(device = torch.device(dev),dtype=torch.float)
            y = y.to(device = torch.device(dev),dtype=torch.float)
            pred = model(x) # Inferring with image x
            loss = lossF(pred, y) # computing loss function
            diceScore = dice(pred, y.int())
            
            opt.zero_grad() # Reset optimizer gradients
            loss.backward() # Backpropagating loss 
            opt.step()      # Optimizing model weights 
            
            # Updating progress display
            totalTrainLoss += loss ; totalTrainDice += diceScore
            trainLoop.set_description(
                "Epoch [{}/{}] -- loss = {:.3f} - F1 = {:.3f}"
                .format(e+1, nEp, loss, diceScore)
            )
        
            
        # Validation
        with torch.no_grad() : # No weights updating
            model.eval() # Testing model (validation)
            
            # Validation progress bar set up 
            tqdm._instances.clear() 
            testLoop = tqdm(vLoad, position=0, leave=True)
            
            for (_, (x, y, n)) in enumerate(testLoop) :
                x = x.to(device = torch.device(dev),dtype=torch.float)
                y = y.to(device = torch.device(dev),dtype=torch.float)
                pred = model(x) # Inferring 
                loss, diceScore = lossF(pred, y), dice(pred, y.int())
                
                totalTestLoss += loss ; totalTestDice += diceScore 
                testLoop.set_description(
                    "Validation  -- loss = {:.3f} - F1 = {:.3f}"
                    .format(loss, diceScore)
                )
            sch.step(totalTestLoss) # Reducing learning rate if Plateau
            
        avgTrainLoss, avgTestLoss = totalTrainLoss/tStep, totalTestLoss/sStep
        avgTrainDice, avgTestDice = totalTrainDice/tStep, totalTestDice/sStep
        
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["valid_loss"].append(avgTestLoss.cpu().detach().numpy())
        H["train_dice"].append(avgTrainDice.cpu().detach().numpy())
        H["valid_dice"].append(avgTestDice.cpu().detach().numpy())
        
        if plot : # If plot enabled : plotting validation results
            img  = x.permute(0, 2, 3, 1)[0]
            segm = pred.permute(0, 2, 3, 1)[0]
            gt   = y.permute(0, 2, 3, 1)[0]
            live_plot(H, img.cpu().detach().numpy(), 
                      segm.cpu().detach().numpy(), 
                      gt.cpu().detach().numpy())
            
        print("Current learning rate :", 
              opt.state_dict()["param_groups"][0]['lr']
        )
        print("train loss : {:.5f} --- validation loss : {:.5f}".format(
                avgTrainLoss, avgTestLoss))
        print("train dice : {:.5f} --- validation dice : {:.5f}\n".format(
                avgTrainDice, avgTestDice))
        
        if((e % ckpStep == 0) or (e == nEp -1)) : 
            now = datetime.now()   
            dtStr = now.strftime("%d%m%Y_%H%M%S")
            checkpoint = {"state_dict": model.state_dict(), 
                          "optimizer":opt.state_dict()
            }
            f = f"Checkpoints/{model.name}_{lossF.name}_epoch{e+1}_{dtStr}.pth"
            save_checkpoints(checkpoint, f)
    
    # End of training loop
    endTime = time.time()-startTime
    print("[END] - Training ended after {}".format(
        format_seconds_to_hhmmss(endTime))
    )
    
    return H, model, f