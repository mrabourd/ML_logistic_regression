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
        x.size == 0:
        return None
    sigmoid = 1 / (1 + np.exp(-1 * x))
    return sigmoid


def logistic_predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * n.
    theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
    y_hat as a numpy.ndarray, a vector of dimension m * 1.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or\
        not isinstance(theta, np.ndarray) or \
        x.size == 0 or theta.shape[1] != 1:
        return None

    m = x.shape[0]
    n = x.shape[1]

    x_one = np.ones((m, 1))
    x_hat = np.column_stack((x_one, x))

    y_hat = np.zeros((m, ))

    pred = x_hat @ theta
    y_hat = sigmoid_(pred)
    return y_hat