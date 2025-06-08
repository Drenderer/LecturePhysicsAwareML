import math
import numpy as np

from jaxtyping import Array
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
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
    ts: Array, qs: Array, fps: int = 30, speedup: int = 3, color=None
) -> HTML:
    """Animate one or multiple solutions of the spring pendulum via
    matplotlib in a Jupyter notebook.

    Args:
        ts: 1D array of monotonically increasing time stamps
            with ``shape=(k,)``.
        qs: 2D or 3D array of ``shape=(..., k, n)`` containing the x
            and y coordinates of pendulum mass as the first two
            entries along the last axis. If 3D then the first axis is
            the batch axis.
        fps: Frames per second. Defaults to 30.
        speedup: Speedup of the animation. Defaults to 3.
        color: Color of the pendulum bob(s). Defaults to None, which uses blue for the first
            pendulum bob and gray for all others.

    Returns:
        HTML display object. When this object is returned by an
        expression or passed to the display function of a jupyter notebook,
        it will result in the data being displayed in the frontend.
    """
    qs = qs if qs.shape[-1] == 2 else qs[..., :2]

    qs = qs if qs.ndim == 3 else qs[np.newaxis, ...]

    fig, ax = plt.subplots()
    x_max, y_max = np.maximum(np.abs(qs).max(axis=(0, 1)), 0.5)

    if color is None:
        bob_colors = ["blue"] + ["gray"] * (qs.shape[0] - 1)
        spring_colors = ["black"] + ["gray"] * (qs.shape[0] - 1)
    else:
        bob_colors = [color] * qs.shape[0]
        spring_colors = ["gray"] * qs.shape[0]

    artists_per_q = []
    for batch_index, (q, bob_color, spring_color) in enumerate(zip(qs, bob_colors, spring_colors)):
        ax.set(
            xlim=1.1 * np.array([-x_max, x_max]),
            ylim=1.1 * np.array([-y_max, 0.3]),
            xlabel="x",
            ylabel="y",
        )
        ax.set_aspect("equal")
        (spring,) = ax.plot(
            *make_spring(np.array([0.0, 0.0]), q[0], 50, 0.1),
            c=spring_color,
            zorder=3 if batch_index == 0 else 1,
            lw=1,
        )
        bob = ax.scatter(
            q[0, 0],
            q[0, 1],
            s=500,
            marker="o",
            c=bob_color,
            zorder=4 if batch_index == 0 else 2,
        )
        artists_per_q.append((spring, bob))
    ax.scatter(
        0, 0, s=50, marker="o", facecolors="white", edgecolors="black", zorder=3
    )

    def animate(t):
        i = np.argmin(np.abs(ts - t))
        for q, (spring, bob) in zip(qs, artists_per_q):
            bob.set_offsets(q[i])
            spring.set_data(*make_spring(np.array([0.0, 0.0]), q[i], 50, 0.1))

    delta_t = ts[-1] - ts[0]
    t_frames = np.linspace(ts[0], ts[-1], int(delta_t * fps / speedup))

    ani = FuncAnimation(fig, animate, frames=t_frames) # type: ignore

    plt.close()
    return HTML(ani.to_jshtml(fps=fps))


def plot_trajectory(ts: Array, ys: Array, ax:Axes|None=None, **kwargs) -> Axes:
    """Plot the trajectory of a spring pendulum.

    Args:
        ts: 1D array of monotonically increasing time stamps
            with ``shape=(k,)``.
        ys: 2D array of ``shape=(k, n)`` containing the x and y
            coordinates of pendulum mass as the first two entries along
            the last axis.
    """
    if ys.ndim != 2 or ys.shape[-1] < 2:
        raise ValueError("ys must be a 2D array with at least 2 columns.")

    if ax is None:
        ax = plt.gca()

    label = kwargs.pop("label", "")
    ax.plot(ts, ys[:, 0], label=label+" $q_x$", ls="-", **kwargs)
    ax.plot(ts, ys[:, 1], label=label+" $q_y$", ls="-.", **kwargs)
    ax.set(
        xlabel="Time t",
        ylabel="Position $q$",
        title="Spring Pendulum Position Over Time",
    )
    ax.legend()
    return ax