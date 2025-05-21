import numpy as np
import matplotlib.pyplot as plt



i_range = 10

colorsCPS = np.array([[(194/255, 76/255, 76/255)],
                   [(246/255, 163/255, 21/255)],
                   [(67/255, 83/255, 132/255)],
                   [(22/255, 164/255, 138/255)],
                   [(187/255, 187/255, 187/255)]])

colorsCPS=np.concatenate([colorsCPS,colorsCPS],axis=0)
colorsCPS = np.tile(colorsCPS,[3,1,1])
    

def plot_spectra(wn, spectra, color, title):
    
    fig, ax = plt.subplots(1, dpi=80,figsize=(6,4))
    
    
    for i in range(100):
        ax.plot(wn[:],spectra[i,:],color=colorsCPS[color],alpha=0.2)         
      
    ax.set_xticks([400,1100,1800])
    ax.set_yticks([0,1])
        
    
    plt.title(f'Bacterium {title}')
    plt.show()
    
    
def plot_latent_space_ij(model_D, spectra, label, i, j):
    
    ls = model_D(spectra)
    
    fig, ax = plt.subplots(1, dpi=80,figsize=(6,6))
        
    ax.scatter(ls[:,i], ls[:,j],c=colorsCPS[label[:].astype(int)], alpha=0.8,s=50)
        
    ax.legend()
    ax.set_xlabel(f'$h_{i+1}$')
    ax.set_ylabel(f'$h_{j+1}$')    
    
    plt.locator_params(nbins=3)         
    plt.show()



def plot_loss(h):
    
    plt.figure()
    plt.semilogy(h.history['loss'], label='training loss')
    plt.grid(which='both')
    plt.xlabel('calibration epoch')
    plt.ylabel('log$_{10}$ MSE')
    plt.legend()
    plt.show()
