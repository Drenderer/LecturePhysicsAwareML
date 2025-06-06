import equinox as eqx
import jax.numpy as jnp
from typing import Literal

class SpringPendulumDerivative(eqx.Module):
    """
    Derivative function (i.e., the right-hand side of the ODE) for the
    pring pendulum system in cartesian or polar coordinates.
    This computes the derivatives of the state variables given the current state.
    """

    k: float  = eqx.field(static=True)  #: Spring constant
    m: float  = eqx.field(static=True) #: Mass of the pendulum bob
    g: float  = eqx.field(static=True) #: Gravitational acceleration
    l0: float = eqx.field(static=True) #: Natural length of the spring
    c: float  = eqx.field(static=True) #: Damping coefficient
    coordinates: Literal['cartesian', 'polar'] #: Coordinate system used ('cartesian' or 'polar')

    def __init__(self, k: float, m: float, g: float, l0: float, c: float=0.0, coordinates: Literal['cartesian', 'polar'] = 'cartesian'):
        self.k = k
        self.m = m
        self.g = g
        self.l0 = l0
        self.c = c
        self.coordinates = coordinates

    def cartesian_derivative(self, t, y, u):
        qx, qy, vx, vy = y
        length = jnp.sqrt(qx**2 + qy**2)
        qx_t = vx
        qy_t = vy
        vx_t = -self.k / self.m * (1 - self.l0 / length) * qx
        vy_t = -self.k / self.m * (1 - self.l0 / length) * qy - self.g
        return jnp.stack([qx_t, qy_t, vx_t, vy_t])
    
    def polar_derivative(self, t, y, u):
        r, θ, vr, vθ = y
        r_t = vr
        θ_t = vθ
        vr_t = -(self.k / self.m) * (r - self.l0) + self.g * jnp.cos(θ) + r * vθ**2
        vθ_t = -(self.g / r) * jnp.sin(θ) - 2 * vr * vθ / r
        return jnp.stack([r_t, θ_t, vr_t, vθ_t])

    def __call__(self, t, y, u):
        if self.coordinates == 'cartesian':
            return self.cartesian_derivative(t, y, u)
        elif self.coordinates == 'polar':
            return self.polar_derivative(t, y, u)
        else:
            raise ValueError("Invalid coordinate system. Use 'cartesian' or 'polar'.")