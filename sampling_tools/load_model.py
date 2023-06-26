import torch
import torch.nn as nn
import numpy as np
from sampling_tools.spectral_normalize_chen import spectral_norm

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(spectral_norm(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(spectral_norm(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(spectral_norm(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False)))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out

# ---- load the model based on the type and sigma (noise level) ---- 
def load_model(model_type, sigma,device):

    path = "Pretrained_models/" + model_type + "_noise" + str(sigma) + ".pth"

    net = DnCNN(channels=1, num_of_layers=17)
    model = nn.DataParallel(net).cuda(device)
    
    model.load_state_dict(torch.load(path))
    model.eval()
    
    return model