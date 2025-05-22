"""Load data. """
import numpy as np
import tensorflow as tf

I_RANGE = 100
N_SPECTRA = 50


def load_data(cases):
    n_cases = len(cases)
    wn = np.load("./LecturePhysicsAwareML/Autoencoder/data/wavenumbers.npy")
    wn_c = np.tile(np.expand_dims(wn, axis=0), (n_cases * N_SPECTRA, 1))
    X_finetune = np.load("./LecturePhysicsAwareML/Autoencoder/data/X_finetune.npy")

    spectra = []
    label = []
    for i in range(n_cases):
        case = cases[i]
        spectra2 = X_finetune[case * I_RANGE : case * I_RANGE + I_RANGE, :]
        label2 = i * np.ones_like(spectra2[:, 0])
        spectra.append(spectra2[0:N_SPECTRA, :])
        label.append(label2[0:N_SPECTRA])
    spectra_c = tf.concat(spectra, 0).numpy()
    label_c = tf.concat(label, 0).numpy()

    return wn_c, spectra_c, label_c


def load_single_case(case):
    wn = np.load("./LecturePhysicsAwareML/Autoencoder/data/wavenumbers.npy")
    X_finetune = np.load("./LecturePhysicsAwareML/Autoencoder/data/X_finetune.npy")

    spectra = X_finetune[case * I_RANGE : case * I_RANGE + I_RANGE, :]
    spectra = spectra[0:N_SPECTRA, :]

    return wn, spectra
