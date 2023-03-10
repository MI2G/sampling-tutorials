"""
Created on Mon Feb 15 11:10:03 2021

@author: s1737876
"""
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cshift(x,L):

    with torch.no_grad():

        N = len(x)
        y = torch.zeros(N)
        
        if L == 0:
            y = x.clone().detach()
            return y
        
        if L > 0:
            y[L:] = x[0:N-L]
            y[0:L] = x[N-L:N]
        else:
            L=int(-L)
            y[0:N-L] = x[L:N]
            y[N-L:N] = x[0:L]
            
        return y           