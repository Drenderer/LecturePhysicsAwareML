import equinox as eqx
import jax.numpy as jnp
from typing import Literal
from jaxtyping import Array

# TODO: Add dissipation


class SpringPendulum(eqx.Module):
    """
    Derivative function (i.e., the right-hand side of the ODE) for the
    pring pendulum system in cartesian or polar coordinates.
    This computes the derivatives of the state variables given the current state.
    """

    k: float = eqx.field(static=True)  #: Spring constant
    m: float = eqx.field(static=True)  #: Mass of the pendulum bob
    g: float = eqx.field(static=True)  #: Gravitational acceleration
    l0: float = eqx.field(static=True)  #: Natural length of the spring
    c: float = eqx.field(static=True)  #: Damping coefficient
    coordinates: Literal[
        "cartesian", "polar"
    ]  #: Coordinate system used ('cartesian' or 'polar')

    def __init__(
        self,
        k: float,
        m: float,
        g: float,
        l0: float,
        c: float = 0.0,
        coordinates: Literal["cartesian", "polar"] = "cartesian",
    ):
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

    def __call__(self, t: Array | None, y: Array, u: Array | None = None) -> Array:
        if self.coordinates == "cartesian":
            return self.cartesian_derivative(t, y, u)
        elif self.coordinates == "polar":
            return self.polar_derivative(t, y, u)
        else:
            raise ValueError("Invalid coordinate system. Use 'cartesian' or 'polar'.")
        
    def compute_energy(self, ys: Array) -> Array:
        """
        Compute the total mechanical energy of the spring pendulum system.

        Args:
            ys: State array with ``shape (k, n)``, where ``n`` is the number of 
                state variables and ``k`` is the number of time stamps.

        Returns:
            Array of total mechanical energy at each time step.
        """
        assert ys.ndim == 2, "State array must be 2D with shape (num_time, 4)."

        if self.coordinates == "cartesian":
            qx, qy, vx, vy = jnp.split(ys, 4, axis=-1)
            potential_energy = self.m * self.g * qy
            kinetic_energy = 0.5 * self.m * (vx**2 + vy**2)
            spring_energy = 0.5 * self.k * ((qx**2 + qy**2)**0.5 - self.l0)**2
        elif self.coordinates == "polar":
            r, θ, vr, vθ = jnp.split(ys, 4, axis=-1)
            potential_energy = - self.m * self.g * r * jnp.cos(θ)
            kinetic_energy = 0.5 * self.m * (vr**2 + (r * vθ)**2)
            spring_energy = 0.5 * self.k * (r - self.l0)**2
        else:
            raise ValueError("Invalid coordinate system. Use 'cartesian' or 'polar'.")

        total_energy = potential_energy + kinetic_energy + spring_energy
        return jnp.squeeze(total_energy, axis=-1)
