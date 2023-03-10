from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

def to_numpy(x):
    return x.detach().cpu().numpy().squeeze()

def NRMSE(x, y):
    x_np = to_numpy(x)
    return np.linalg.norm(x_np - to_numpy(y),'fro')/np.linalg.norm(x_np,'fro')

def SSIM(x, y):
    return ssim(to_numpy(x), to_numpy(y), data_range=1)

def PSNR(x, y):
    return psnr(to_numpy(x), to_numpy(y), data_range=1)

