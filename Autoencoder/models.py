"""Neural network models."""
import tensorflow as tf
from tensorflow.keras import layers


class MLP(layers.Layer):
    """A simple multi-layer perceptron."""

    def __init__(self, units, activation):
        super().__init__()

        self.ls = []
        for u, a in zip(units, activation):
            self.ls += [layers.Dense(u, a)]

    def __call__(self, x):
        for layer in self.ls:
            x = layer(x)
        return x


def build(input_shape, **kwargs):
    x = tf.keras.Input(shape=[input_shape])
    y = MLP(**kwargs)(x)
    model = tf.keras.Model(inputs=[x], outputs=[y])
    model.compile("adam", "mse")
    return model
