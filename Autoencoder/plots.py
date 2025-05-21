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
    
    fig, ax = plt.subplots(1, dpi=500,figsize=(6,4))
    
    
    for i in range(100):
        ax.plot(wn[:],spectra[i,:],color=colorsCPS[color],alpha=0.2)         
      
    ax.set_xticks([400,1100,1800])
    ax.set_yticks([0,1])
        
    
    plt.title(f'Bacterium {title}')
    plt.show()
    
    
def plot_latent_space_ij(model_D, spectra_c, spectra_t, label_c, label_t, i, j):
    
    ls_c = model_D(spectra_c)
    ls_t = model_D(spectra_t)
    
    fig, ax = plt.subplots(1, dpi=500,figsize=(6,6))
    
    ax.scatter(ls_c[0:1,i], ls_c[0:1,j],c=colorsCPS[4], alpha=0.8,label='calibration',s=50)
    ax.scatter(ls_t[0:1,i], ls_t[0:1,j],c=colorsCPS[4], alpha=0.8,marker='s',edgecolors= "black",label='test',s=50)
        
    ax.scatter(ls_c[:,i], ls_c[:,j],c=colorsCPS[label_c[:].astype(int)], alpha=0.8,s=50)
    ax.scatter(ls_t[:,i], ls_t[:,j],c=colorsCPS[label_t[:].astype(int)], alpha=0.8,marker='s',edgecolors= "black",s=50)
    
        
    ax.legend()
    ax.set_xlabel(f'$h_{i+1}$')
    ax.set_ylabel(f'$h_{j+1}$')    
    
    plt.locator_params(nbins=3)         
    plt.show()



def plot_loss(h):
    
    plt.figure()
    plt.semilogy(h.history['loss'], label='training loss')
    plt.semilogy(h.history['val_loss'], label='validation loss')
    plt.grid(which='both')
    plt.xlabel('calibration epoch')
    plt.ylabel('log$_{10}$ MSE')
    plt.legend()
    plt.show()
