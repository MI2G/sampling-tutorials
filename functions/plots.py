import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.measure import block_reduce
from .measures import PSNR, SSIM, NRMSE


def plot_im(x, title="Title"):

    plt.subplots()
    if torch.is_tensor(x):
        plt.imshow(x.detach().cpu().numpy(), cmap="gray")
    else:
        plt.imshow(x, cmap="gray")
    plt.colorbar()
    plt.tight_layout()
    plt.axis('off')
    plt.title(title)
        

def plot_trace(x, title="Title", x_label="xlabel"):
    plt.subplots()
    if torch.is_tensor(x):
        plt.plot(x.detach().cpu().numpy())
    else:
        plt.plot(x)
    plt.tight_layout()
    plt.title(title)
    plt.xlabel(x_label)


def plots(x,y,post_meanvar,post_meanvar_absfourier, nrmse_values, psnr_values, ssim_values, logPi_trace):
    
    post_mean_numpy = post_meanvar.get_mean().detach().cpu().numpy()
    post_var_numpy = post_meanvar.get_var().detach().cpu().numpy()
    
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize = (15,10))
    fig.tight_layout(pad=.01)
    
    # --- Ground truth
    plot1 = plt.figure()
    axes[0,0].imshow(x.detach().cpu().numpy(), cmap="gray")
    axes[0,0].set_title('Ground truth image')
    axes[0,0].axis('off')
    plt.close()

    # --- Blurred
    plot1 = plt.figure()
    axes[0,1].imshow(y.detach().cpu().numpy(), cmap="gray")
    axes[0,1].set_title('Blurred noisy image')
    axes[0,1].axis('off')
    plt.close()

    # --- MMSE
    plot1 = plt.figure()
    axes[0,2].imshow(post_mean_numpy, cmap="gray")
    axes[0,2].set_title('x - posterior mean')
    axes[0,2].axis('off')

    # --- Variance
    axes[0,3].imshow(post_var_numpy, cmap="gray")
    axes[0,3].set_title('x - posterior variance')
    axes[0,3].axis('off')
    plt.close()

    # --- MMSE / Var
    plot1 = plt.figure()
    axes[1,0].imshow(post_mean_numpy/np.sqrt(post_meanvar.get_var().detach().cpu().numpy()), cmap="gray")
    axes[1,0].set_title('x - posterior mean/posterior SD')
    axes[1,0].axis('off')
    plt.close()

    # --- Var / MMSE
    plot1 = plt.figure()
    axes[1,1].imshow(np.sqrt(post_var_numpy)/post_mean_numpy,cmap="gray")
    axes[1,1].set_title('x - Coefs of variation')
    axes[1,1].axis('off')
    plt.close()

    # --- Mean Fourier coefs
    plot1 = plt.figure()
    axes[1,2].imshow(torch.log(post_meanvar_absfourier.get_mean()).detach().cpu().numpy())
    axes[1,2].set_title('Mean coefs (log-scale)')
    axes[1,2].axis('off')
    plt.close()
    
    # --- Variance Fourier coefs
    plot1 = plt.figure()
    axes[1,3].imshow(torch.log(post_meanvar_absfourier.get_var()).detach().cpu().numpy())
    axes[1,3].set_title('Var coefs (log-scale)')
    axes[1,3].axis('off')
    plt.close()
                
    # --- NRMSE ---                
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (15,5))
    fig.tight_layout(pad=.01)
    
    plot1 = plt.figure()
    axes[0].plot(np.arange(len(nrmse_values))[::10], nrmse_values[::10], label =  "-- NRMSE --")
    axes[0].set_title('NRMSE of $X$ vs $x_{gr}$')
    axes[0].legend()
    plt.close()
    
    # --- PSNR ---
    plot1 = plt.figure()
    axes[1].plot(np.arange(len(psnr_values))[::10], psnr_values[::10], label =  "-- PSNR --")
    axes[1].set_title('PSNR of $X$ vs $x_{gr}$')
    axes[1].legend()
    plt.close()

    # --- SSIM ---
    plot1 = plt.figure()
    axes[2].plot(np.arange(len(ssim_values))[::10],ssim_values[::10], label =  "-- SSIM --")
    axes[2].set_title('SSIM of $X$ vs $x_{gr}$')
    axes[2].legend()
    plt.close()
                     
    # --- log pi
    plot = plt.figure(figsize = (15,10))
    
    plt.plot(np.arange(len(logPi_trace))[::10],logPi_trace[::10], label =  "- $\log \pi$ -")
    plt.legend()
    plt.show()
    plt.close()

def plots_libary(true_x, y, mean_x, var_x, logPi_trace, samples, n_iter):

    if not isinstance(samples, torch.Tensor):
        samples = torch.tensor(samples)
    
    mean_ = mean_x.detach().cpu().squeeze()
    var_ = var_x.detach().cpu().squeeze()
    
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize = (15,10))
    fig.tight_layout(pad=.01)
    
    # --- Ground truth
    axes[0,0].imshow(true_x.detach().cpu().squeeze(), cmap="gray")
    axes[0,0].set_title('Ground truth image')
    axes[0,0].axis('off')

    # --- Blurred
    axes[0,1].imshow(y.detach().cpu().squeeze(), cmap="gray")
    axes[0,1].set_title('Blurred noisy image')
    axes[0,1].axis('off')

    # --- MMSE
    axes[0,2].imshow(mean_, cmap="gray")
    axes[0,2].set_title('x - posterior mean')
    axes[0,2].axis('off')

    # --- Variance
    axes[0,3].imshow(var_, cmap="gray")
    axes[0,3].set_title('x - posterior variance')
    axes[0,3].axis('off')

    # --- MMSE / SD
    axes[1,0].imshow(mean_/np.sqrt(var_), cmap="gray")
    axes[1,0].set_title('x - posterior mean/posterior SD')
    axes[1,0].axis('off')

    # --- SD / MMSE
    axes[1,1].imshow(np.sqrt(var_)/mean_,cmap="gray")
    axes[1,1].set_title('x - Coefs of variation (SD/mean)')
    axes[1,1].axis('off')

    # --- Mean Fourier coefs
    axes[1,2].imshow(np.log(np.abs(np.fft.fft2(mean_))))
    axes[1,2].set_title('Mean coefs fourier space (log-scale)')
    axes[1,2].axis('off')
    
    # --- Variance Fourier coefs
    axes[1,3].imshow(np.log(np.abs(np.fft.fft2(var_))))
    axes[1,3].set_title('Var coefs fourier space (log-scale)')
    axes[1,3].axis('off')
                
    # --- NRMSE ---                
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (15,5))
    fig.tight_layout(pad=.01)
    
    nrmse_values = np.zeros(samples.shape[0])
    psnr_values = np.zeros(samples.shape[0])
    ssim_values = np.zeros(samples.shape[0])

    # initialize welford algorithm to compute quality measures
    # from cumulative mean
    from functions.welford import welford
    post_meanvar = welford(samples[0])

    for i in range(samples.shape[0]):
        # update the sample summary statistics
        post_meanvar.update(samples[i])
        # compute the quality measures
        running_mean = post_meanvar.get_mean().detach().cpu().squeeze()
        nrmse_values[i] = NRMSE(running_mean, true_x)
        psnr_values[i] = PSNR(running_mean, true_x)
        ssim_values[i] = SSIM(running_mean, true_x)

    axes[0].plot(np.linspace(1,n_iter,samples.shape[0]), nrmse_values, label =  "-- NRMSE --")
    axes[0].set_title('NRMSE of $X$ vs $x_{gr}$')
    axes[0].legend()
    
    # --- PSNR ---
    axes[1].plot(np.linspace(1,n_iter,samples.shape[0]), psnr_values, label =  "-- PSNR --")
    axes[1].set_title('PSNR of $X$ vs $x_{gr}$')
    axes[1].legend()

    # --- SSIM ---
    axes[2].plot(np.linspace(1,n_iter,samples.shape[0]), ssim_values, label =  "-- SSIM --")
    axes[2].set_title('SSIM of $X$ vs $x_{gr}$')
    axes[2].legend()
                     
    # --- log pi
    plt.figure(figsize = (15,10))
    plt.semilogx(np.arange(1,logPi_trace.shape[0]+1), logPi_trace, label =  "- $\log \pi$ -")
    plt.legend()
    plt.show()
    
    
    
def downsampling_variance(X_chain):

    scale = [1,2,4,8]
    n_samples = X_chain.shape[0]
    nx = X_chain.shape[1]
    ny = X_chain.shape[2]
 
    st_deviation_down= []

    for k,i in enumerate(scale):
        
        downsample_array= np.zeros([n_samples,int(nx/(i*2)),int(ny/(i*2))])

        st_deviation_down.append(np.zeros([int(nx/(i*2)),int(ny/(i*2))]))

        for j in range(n_samples):

            downsample_array[j]= block_reduce(X_chain[j], block_size=(i*2,i*2), func=np.mean)

        meanSample_down= np.mean(downsample_array,0)

        second_moment_down= np.mean(downsample_array**2,0)

        st_deviation_down[k] = np.sqrt(second_moment_down - meanSample_down**2)	

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    _, axes = plt.subplots(nrows=1, ncols=4, figsize = (15,20))
    im=axes[0].imshow(st_deviation_down[0],cmap="gray")
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    axes[0].axis('off') 
    
    im=axes[1].imshow(st_deviation_down[1],cmap="gray")
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)    
    plt.colorbar(im, cax=cax)
    axes[1].axis('off') 
    
    im=axes[2].imshow(st_deviation_down[2],cmap="gray")
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax) 
    axes[2].axis('off') 
    
    im=axes[3].imshow(st_deviation_down[3],cmap="gray")
    divider = make_axes_locatable(axes[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)  
    axes[3].axis('off') 
