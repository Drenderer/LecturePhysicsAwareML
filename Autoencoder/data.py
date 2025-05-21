import numpy as np
import tensorflow as tf
import random

i_range = 100

# %%   
"""
Load calibration data and plot them

"""

def load_data(cases,n_calib,n_test):
    
    n_cases = len(cases)

    paths_all = set(range(i_range*n_cases))
    paths_c = set(random.sample(paths_all, n_calib))
    paths_all -= paths_c
    paths_t = set(random.sample(paths_all,n_test))
    paths_all -= paths_t
    paths_v = paths_all
    
    paths_c = list(paths_c)
    paths_t = list(paths_t)
    paths_v = list(paths_v)
    
    wn = np.load('./data/wavenumbers.npy')
    
    wn_c = np.tile(np.expand_dims(wn,axis=0),(n_cases*70,1))
    wn_t = np.tile(np.expand_dims(wn,axis=0),(n_cases*15,1))
    wn_v = np.tile(np.expand_dims(wn,axis=0),(n_cases*15,1))
    
    X_finetune = np.load('./data/X_finetune.npy')
    
    spectra = []
    label = []
    for i in range(n_cases):
        case = cases[i]
        spectra2 = X_finetune[case*i_range:case*i_range+i_range,:]
        label2 = i*np.ones_like(spectra2[:,0]) 
        spectra.append(spectra2)
        label.append(label2)
    spectra = tf.concat(spectra,0).numpy()
    label = tf.concat(label,0).numpy()
        
    spectra_c = spectra[np.array(paths_c),:]
    spectra_t = spectra[np.array(paths_t),:]
    spectra_v = spectra[np.array(paths_v),:]
    
    label_c = label[np.array(paths_c)]
    label_t = label[np.array(paths_t)]
    label_v = label[np.array(paths_v)]
    
    return wn_c, wn_t, wn_v, spectra_c, spectra_t, spectra_v, label_c, label_t, label_v


def load_single_case(case):
    
    wn = np.load('./data/wavenumbers.npy')    
    X_finetune = np.load('./data/X_finetune.npy')

    spectra = X_finetune[case*i_range:case*i_range+i_range,:]
    
    
    return wn, spectra

    





