# -*- coding: utf-8 -*-
"""
Created on Wed May 19 20:40:43 2021

@author: SavvasM
"""
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Grad_Image(x):

    with torch.no_grad():

        x = x.to(device).clone()
        x_temp = x[1:, :] - x[0:-1,:]
        dux = torch.cat((x_temp.T,torch.zeros(x_temp.shape[1],1,device=device)),1).to(device)
        dux = dux.T
        x_temp = x[:,1:] - x[:,0:-1]
        duy = torch.cat((x_temp,torch.zeros((x_temp.shape[0],1),device=device)),1).to(device)
        return  torch.cat((dux,duy),dim=0).to(device)
