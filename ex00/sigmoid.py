import numpy as np


def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
    x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
    The sigmoid value as a numpy.ndarray of shape (m, 1).
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or \
        x.shape[1] != 1 or x.size == 0:
        return None
    sigmoid = 1 / (1 + np.exp(-1 * x))
    return sigmoid