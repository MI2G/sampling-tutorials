#    Loads the neural network to be used within the Pnp-ULA algorithm

#    The network and source code was developed by the authors of this paper
#    E. K. Ryu, J. Liu, S. Wang, X. Chen, Z. Wang, and W. Yin. 
#    "Plug-and-Play Methods Provably Converge with Properly Trained Denoisers." ICML, 2019.

#    GitHub account https://github.com/uclaopt/Provable_Plug_and_Play/

#    (minorly) adapted in pytorch by: MI2G
#    Copyright (C) 2023 MI2G
#    Dobson, Paul pdobson@ed.ac.uk
#    Kemajou, Mbakam Charlesquin cmk2000@hw.ac.uk
#    Klatzer, Teresa t.klatzer@sms.ed.ac.uk
#    Melidonis, Savvas sm2041@hw.ac.uk
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
