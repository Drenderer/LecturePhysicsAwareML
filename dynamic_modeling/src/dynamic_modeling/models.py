import klax
import jax
import equinox as eqx

from jaxtyping import PRNGKeyArray, Array
from typing import Literal

ACTIVATIONS = dict(
    softplus=jax.nn.softplus,
    relu=jax.nn.relu,
    sigmoid=jax.nn.sigmoid,
)


class NODEDerivative(eqx.Module):
    mlp: klax.nn.MLP

    def __init__(
        self,
        *,
        hidden_layer_sizes: list[int] = [16, 16],
        activation: Literal["softplus", "relu", "sigmoid"] = "softplus",
        key: PRNGKeyArray,
    ):
        self.mlp = klax.nn.MLP(
            in_size=4,
            out_size=4,
            width_sizes=hidden_layer_sizes,
            weight_init=jax.nn.initializers.variance_scaling(
                0.2, mode="fan_avg", distribution="truncated_normal"
            ),
            activation=ACTIVATIONS[activation],
            key=key,
        )

    def __call__(self, t: Array, y: Array, u: Array | None = None) -> Array:
        return self.mlp(y)
