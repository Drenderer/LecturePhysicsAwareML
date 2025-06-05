from dynamic_modeling import ODESolver
import klax

import equinox as eqx
from jaxtyping import Array
import jax
import jax.numpy as jnp
from jax import random as jr
from jax.nn.initializers import variance_scaling

import matplotlib.pyplot as plt


class Derivative(eqx.Module):
    """
    Derivative function of a linear two mass oscillator system, with a
    force input u acting on the second mass.
    """

    A: Array
    B: Array
    J: Array
    R: Array
    Q: Array

    def __init__(self, m1, m2, k1, k2, d1, d2):
        zeros = jnp.zeros((2, 2))

        # Structure matrix
        mass = jnp.array([[m1, 0], [0, m2]])
        mass_inv = jnp.linalg.inv(mass)
        J = jnp.block([[zeros, mass_inv], [-mass_inv, zeros]])
        self.J = J

        # Resistive matrix
        diss = jnp.array(
            [
                [(d1 + d2) / (m1 * m1), -d2 / (m1 * m2)],
                [-d2 / (m1 * m2), d2 / (m1 * m2)],
            ]
        )
        R = jnp.block([[zeros, zeros], [zeros, diss]])
        self.R = R

        # Hamililtonian quadratic form H=xQx
        Q = jnp.array(
            [[k1 + k2, -k2, 0, 0], [-k2, k2, 0, 0], [0, 0, m1, 0], [0, 0, 0, m2]]
        )
        self.Q = Q

        self.A = (J - R) @ Q

        # Input matrix
        self.B = jnp.array([0, 0, 0, 1 / m2])[:, None]

    def __call__(self, t, y, u):
        return self.A @ y + self.B @ u