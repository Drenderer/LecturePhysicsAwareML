"""Plot data and latent space."""

import numpy as np
import matplotlib.pyplot as plt


I_RANGE = 10
N_SPECTRA = 50

CPS_COLORS = np.array(
    [
        [(194 / 255, 76 / 255, 76 / 255)],
        [(246 / 255, 163 / 255, 21 / 255)],
        [(67 / 255, 83 / 255, 132 / 255)],
        [(22 / 255, 164 / 255, 138 / 255)],
        [(187 / 255, 187 / 255, 187 / 255)],
    ]
)
CPS_COLORS = np.concatenate([CPS_COLORS, CPS_COLORS], axis=0)
CPS_COLORS = np.tile(CPS_COLORS, [3, 1, 1])


def plot_spectra(raman_shift, intensity, color, title):
    _, ax = plt.subplots(1, dpi=80, figsize=(6, 4))
    for i in range(N_SPECTRA):
        ax.plot(raman_shift[:], intensity[i, :], color=CPS_COLORS[color], alpha=0.2)

    ax.set_xticks([400, 1100, 1800])
    ax.set_yticks([0, 1])
    ax.set_ylabel("Intensity")
    ax.set_xlabel("Raman shift [cm$^{-1}$]")

    plt.title(f"Bacterium {title}")
    plt.show()


def plot_latent_space_ij(encoder, intensity, label, i, j):
    latent_variables = encoder(intensity)

    _, ax = plt.subplots(1, dpi=80, figsize=(6, 6))
    ax.scatter(
        latent_variables[:, i],
        latent_variables[:, j],
        c=CPS_COLORS[label[:].astype(int)],
        alpha=0.8,
        s=50,
    )
    ax.legend()
    ax.set_xlabel(f"$h_{i + 1}$")
    ax.set_ylabel(f"$h_{j + 1}$")

    plt.locator_params(nbins=3)
    plt.show()


def plot_loss(h):
    plt.figure()
    plt.semilogy(h.history["loss"], label="training loss")
    plt.grid(which="both")
    plt.xlabel("calibration epoch")
    plt.ylabel("log$_{10}$ MSE")
    plt.legend()
    plt.show()
