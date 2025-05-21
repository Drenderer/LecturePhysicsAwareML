import numpy as np
import tensorflow as tf
import random

i_range = 100
n_spectra = 30

# %%   
"""
Load calibration data and plot them

"""

def load_data(cases):
    
    n_cases = len(cases)

    wn = np.load('./LecturePhysicsAwareML/Autoencoder/data/wavenumbers.npy')
    
    wn_c = np.tile(np.expand_dims(wn,axis=0),(n_cases*n_spectra,1))
    
    X_finetune = np.load('./LecturePhysicsAwareML/Autoencoder/data/X_finetune.npy')
    
    spectra = []
    label = []
    for i in range(n_cases):
        case = cases[i]
        spectra2 = X_finetune[case*i_range:case*i_range+i_range,:]
        label2 = i*np.ones_like(spectra2[:,0]) 
        spectra.append(spectra2[0:n_spectra,:])
        label.append(label2[0:n_spectra])
    spectra_c = tf.concat(spectra,0).numpy()
    label_c = tf.concat(label,0).numpy()
    
    return wn_c, spectra_c, label_c


def load_single_case(case):
    
    wn = np.load('./LecturePhysicsAwareML/Autoencoder/data/wavenumbers.npy')    
    X_finetune = np.load('./LecturePhysicsAwareML/Autoencoder/data/X_finetune.npy')

    spectra = X_finetune[case*i_range:case*i_range+i_range,:]
    
    spectra = spectra[0:n_spectra,:]
    
    
    return wn, spectra

    





