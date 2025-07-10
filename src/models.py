# src/models.py
import numpy as np
from numba import njit


@njit
def f_xt(x, t):
    """
    Kitagawa model state transition function: x_t = 0.5 * x_{t-1} + 25 * x_{t-1} / (1 + x_{t-1}^2) + 8 * cos(1.2 * t)

    Args:
        x (float): State at time t-1.
        t (int): Current time step t.

    Returns:
        float: State at time t before adding process noise.
    """
    return 0.5 * x + 25 * x / (1 + x ** 2) + 8 * np.cos(1.2 * t)

@njit
def h_zt(x):
    """
    Kitagawa model observation function: z_t = x_t^2 / 20

    Args:
        x (float): True state x_t.

    Returns:
        float: Observation z_t before adding observation noise.
    """
    return x ** 2 / 20


def simulate_kitagawa(T, Q, R, x0, seed=None):
    """
    Simulates the Kitagawa nonlinear system to generate true states and observations.

    Args:
        T (int): Total number of time steps.
        Q (float): Process noise variance.
        R (float): Observation noise variance.
        x0 (float): Initial true state.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple:
            - x_true (numpy.ndarray): Array of true states.
            - y_obs (numpy.ndarray): Array of noisy observations.
    """
    if seed is not None:
        np.random.seed(seed)

    x_true = np.zeros(T)
    y_obs = np.zeros(T)

    x_true[0] = x0 + np.sqrt(Q) * np.random.randn()  # Initial state with some noise
    y_obs[0] = h_zt(x_true[0]) + np.sqrt(R) * np.random.randn()

    for t in range(1, T):
        x_true[t] = f_xt(x_true[t - 1], t) + np.sqrt(Q) * np.random.randn()
        y_obs[t] = h_zt(x_true[t]) + np.sqrt(R) * np.random.randn()

    return x_true, y_obs