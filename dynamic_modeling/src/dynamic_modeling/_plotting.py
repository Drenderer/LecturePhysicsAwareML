import math
import numpy as np

from jaxtyping import Array
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def make_spring(start, end, nodes, width):
    # This function was copied from nrsyed: https://github.com/nrsyed/utilities
    # It is licensed under the GNU GENERAL PUBLIC LICENSE (Version 3, 29 June 2007)
    """
    Return a list of points corresponding to a spring.

    @param r1 (array-like) The (x, y) coordinates of the first endpoint.
    @param r2 (array-like) The (x, y) coordinates of the second endpoint.
    @param nodes (int) The number of spring "nodes" or coils.
    @param width (int or float) The diameter of the spring.
    @return An array of x coordinates and an array of y coordinates.
    """

    # Check that nodes is at least 1.
    nodes = max(int(nodes), 1)

    # Convert to numpy array to account for inputs of different types/shapes.
    start, end = np.array(start).reshape((2,)), np.array(end).reshape((2,))

    # If both points are coincident, return the x and y coords of one of them.
    if (start == end).all():
        return start[0], start[1]

    # Calculate length of spring (distance between endpoints).
    length = np.linalg.norm(np.subtract(end, start))

    # Calculate unit vectors tangent (u_t) and normal (u_t) to spring.
    u_t = np.subtract(end, start) / length
    u_n = np.array([[0, -1], [1, 0]]).dot(u_t)

    # Initialize array of x (row 0) and y (row 1) coords of the nodes+2 points.
    spring_coords = np.zeros((2, nodes + 2))
    spring_coords[:, 0], spring_coords[:, -1] = start, end

    # Check that length is not greater than the total length the spring
    # can extend (otherwise, math domain error will result), and compute the
    # normal distance from the centerline of the spring.
    normal_dist = math.sqrt(max(0, width**2 - (length**2 / nodes**2))) / 2

    # Compute the coordinates of each point (each node).
    for i in range(1, nodes + 1):
        spring_coords[:, i] = (
            start
            + ((length * (2 * i - 1) * u_t) / (2 * nodes))
            + (normal_dist * (-1) ** i * u_n)
        )

    return spring_coords[0, :], spring_coords[1, :]


def animate_spring_pendulum(
    ts: Array, qs: Array, fps: int = 30, speedup: int = 3
) -> HTML:
    qs = qs if qs.shape[-1] == 2 else qs[..., :2]

    fig, ax = plt.subplots()
    x_max, y_max = np.maximum(np.abs(qs).max(axis=0), 0.5)
    ax.set(
        xlim=1.1 * np.array([-x_max, x_max]),
        ylim=1.1 * np.array([-y_max, 0.3]),
        xlabel="x",
        ylabel="y",
    )
    ax.set_aspect("equal")
    (spring,) = ax.plot(
        *make_spring(np.array([0.0, 0.0]), qs[0], 50, 0.1), c="black", lw=1
    )
    bob = ax.scatter(qs[0, 0], qs[0, 1], s=500, marker="o", zorder=3)
    _ = ax.scatter(
        0, 0, s=50, marker="o", facecolors="white", edgecolors="black", zorder=3
    )

    def animate(t):
        i = np.argmin(np.abs(ts - t))
        bob.set_offsets(qs[i])
        spring.set_data(*make_spring(np.array([0.0, 0.0]), qs[i], 50, 0.1))
        return spring, bob

    delta_t = ts[-1] - ts[0]
    t_frames = np.linspace(ts[0], ts[-1], int(delta_t * fps / speedup))

    ani = FuncAnimation(fig, animate, frames=t_frames)

    plt.close()
    return HTML(ani.to_jshtml(fps=fps))