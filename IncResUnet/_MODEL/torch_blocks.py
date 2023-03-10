# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:39:28 2023
@author: tomDaSilva
"""

import torch
import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(self, inCh, outCh, kSize=3, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential( 
            nn.Conv2d(int(inCh), int(outCh), bias=False, 
                      kernel_size=kSize, stride=stride, padding=padding),
            nn.BatchNorm2d(int(outCh)),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)
    
class DoubleConvBlock(nn.Module):

    def __init__(self, inCh, outCh, kSize=3, stride=1, padding=0):
        super(DoubleConvBlock, self).__init__()
        
        self.conv = nn.Sequential( 
            nn.Conv2d(int(inCh), int(outCh), bias=False, 
                      kernel_size=kSize, stride=stride, padding=padding),
            nn.BatchNorm2d(int(outCh)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(outCh), int(outCh), bias=False, 
                      kernel_size=kSize, stride=stride, padding=padding),
            nn.BatchNorm2d(int(outCh)),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)
    
class StemBlock(nn.Module) : 
    
    def __init__(self, inCh, outCh) : 
        
        super(StemBlock, self).__init__()
        
        self.branch0 = nn.Sequential(
            ConvBlock(inCh, outCh//12,  kSize=3),
            ConvBlock(outCh//12, outCh//12,   kSize=3, padding=2),
            ConvBlock(outCh//12, outCh//6, kSize=3, padding=2)
        )
        
        self.branch1A = ConvBlock(outCh//6, outCh//4, kSize=3, stride=1, padding=1)
        self.branch1B = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
        self.branch2A = nn.Sequential(
            ConvBlock(outCh//2.4, outCh//6, kSize=1),
            ConvBlock(outCh//6, outCh//6, kSize=(7, 1), padding=(3, 0)),
            ConvBlock(outCh//6, outCh//6, kSize=(1, 7), padding=(0, 3)),
            ConvBlock(outCh//6, outCh//4, kSize=3, padding=1)
        )
        self.branch2B = nn.Sequential(
            ConvBlock(outCh//2.4, outCh//6, kSize=1),
            ConvBlock(outCh//6, outCh//4, kSize=3, padding=1)
        )

        self.branchpoola = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.branchpoolb = ConvBlock(outCh//2, outCh//2, kSize=3, stride=2, padding=0)

    def forward(self, x):
        x0  = self.branch0(x)
        x1a = self.branch1A(x0)
        x1b = self.branch1B(x0)
        x1  = torch.cat((x1a, x1b), 1)
        x2a = self.branch2A(x1)
        x2b = self.branch2B(x1)
        x2  = torch.cat((x2a, x2b), 1) 
        xpa = self.branchpoola(x2)
        xpb = self.branchpoolb(x2)
        return torch.cat((xpa, xpb), 1)

class InceptionResNetA(nn.Module):
    
    def __init__(self, inCh):

        super(InceptionResNetA, self).__init__()
        self.branch0 = nn.Sequential(
            ConvBlock(inCh, inCh//12, kSize=1),
            ConvBlock(inCh//12, inCh//8, kSize=3, padding=1),
            ConvBlock(inCh//8,  inCh//6, kSize=3, padding=1)
        )

        self.branch1 = nn.Sequential(
            ConvBlock(inCh, inCh//12, kSize=1),
            ConvBlock(inCh//12, inCh//6, kSize=3, padding=1)
        )
        miss = abs((inCh//2) - 3*(inCh//6))
        self.branch2 = ConvBlock(inCh, miss + inCh//6, kSize=1)

        self.reduction = nn.Conv2d(inCh//2, inCh, kernel_size=1)
        self.shortcut = nn.Conv2d(inCh, inCh, kernel_size=1)
        self.out = nn.Sequential(
            nn.BatchNorm2d(inCh),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        residual = torch.cat((x0, x1, x2), 1)
        residual = self.reduction(residual)
        shortcut = self.shortcut(x)
        return self.out(shortcut + residual) 
        
class ReductionA(nn.Module):
    
    def __init__(self, inCh, outCh):
        
        super(ReductionA, self).__init__()
        
        outCh -= inCh
        self.branch0 = nn.Sequential(
            ConvBlock(inCh, outCh//4, kSize=1), 
            ConvBlock(outCh//4, outCh//3, kSize=3), 
            ConvBlock(outCh//3, outCh//2, kSize=3, stride=2, padding=2)
        )
        
        self.branch1 = ConvBlock(inCh, outCh//2, kSize=3, stride=2, padding=1)
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x) :
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        xp = self.branchpool(x)
        return torch.cat((x0, x1, xp), 1)
    
class InceptionResNetB(nn.Module):

    def __init__(self, inCh):

        super(InceptionResNetB, self).__init__()
        
        self.branch0 = nn.Sequential(
            ConvBlock(inCh, inCh//8, kSize=1),
            ConvBlock(inCh//8, inCh//6  , kSize=(1, 7), padding=(0, 3)),
            ConvBlock(inCh//6, inCh//4, kSize=(7, 1), padding=(3, 0))
        )
        
        miss = abs((inCh//2) - 2*(inCh//4))
        self.branch1 = ConvBlock(inCh, miss+ inCh//4, kSize=1)

        self.reduction = nn.Conv2d(int(inCh//2), int(inCh), kernel_size=1)
        self.shortcut = nn.Conv2d(int(inCh), int(inCh), kernel_size=1)
        
        self.out = nn.Sequential(
            nn.BatchNorm2d(int(inCh)),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        residual = torch.cat((x0, x1), 1)
        residual = self.reduction(residual) * 0.1 # Scaling factor [0.1 ; 0.3]
        shortcut = self.shortcut(x)
        return self.out(residual + shortcut)
    
class ReductionB(nn.Module):
    
    def __init__(self, inCh, outCh):
        
        super(ReductionB, self).__init__()
        
        outCh -= inCh
        self.branch0 = nn.Sequential(
            ConvBlock(inCh, outCh//5, kSize=1),
            ConvBlock(outCh//5, outCh//4, kSize=3, stride=2, padding=1)
        )
        
        self.branch1 = nn.Sequential(
            ConvBlock(inCh, outCh//5, kSize=3, padding=1), 
            ConvBlock(outCh//5, outCh//4, kSize=3, stride=2, padding=1)
        )
        
        miss = abs((inCh//2) - 3*(outCh//4))
        self.branch2 = nn.Sequential(
            ConvBlock(inCh, outCh//5, kSize=1), 
            ConvBlock(outCh//5, outCh//4, kSize=3, padding=1), 
            ConvBlock(outCh//4, miss+outCh//4, kSize=3, stride=2, padding=1)
        )
        
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.outCh = inCh + outCh
        
        
    def forward(self, x) : 
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        xp = self.branchpool(x)
        return torch.cat((x0, x1, x2, xp), 1)
    
class InceptionResNetC(nn.Module):
    
    def __init__(self, inCh):
        
        super(InceptionResNetC, self).__init__()
        
        self.branch0 = nn.Sequential(
            ConvBlock(inCh, inCh//8, kSize=1),
            ConvBlock(inCh//8, inCh//7, kSize=(1, 3), padding=(0, 1)),
            ConvBlock(inCh//7, inCh//6, kSize=(3, 1), padding=(1, 0))
        )

        self.branch1 = ConvBlock(inCh, inCh//8, kSize=1)

        self.reduction = nn.Conv2d(int((inCh//8)+(inCh//6)), int(inCh), kernel_size=1)
        self.shortcut = nn.Conv2d(int(inCh), int(inCh), kernel_size=1)

        self.out = nn.Sequential(
            nn.BatchNorm2d(inCh), 
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x) : 
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        residual = torch.cat((x0, x1), 1)
        residual = self.reduction(residual) * 0.1
        shortcut = self.shortcut(x)
        return self.out(residual + shortcut)   
    
class FireSqueeze(nn.Module) : 
    
    def __init__(self, inCh) : 
        super(FireSqueeze, self).__init__() 
        
        self.branch0 = ConvBlock(inCh, inCh//3, kSize=5, padding=2)
        self.branch1 = ConvBlock(inCh, inCh//3, kSize=3, padding=1)
        self.branch2 = ConvBlock(inCh, inCh//3, kSize=1, padding=0)
        self.branch3 = ConvBlock(3*(inCh//3), 3*(inCh//3), kSize=1)
        self.out = ConvBlock(3*(inCh//3), inCh//2, kSize=3, padding=1)
        
    def forward(self, x) : 
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        xc = torch.cat((x0, x1, x2), 1)
        xc = self.branch3(xc)
        return self.out(xc)