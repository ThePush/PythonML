import numpy as np


def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for loop.
    The three arrays must have compatible shapes.

    Args:
    x: has to be a numpy.array, a matrix of shape m * 1.
    y: has to be a numpy.array, a vector of shape m * 1.
    theta: has to be a numpy.array, a 2 * 1 vector.

    Return:
    The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
    None if x, y, or theta is an empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.

    Raises:
    This function should not raise any Exception."""

    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None
    if (
        x.shape != (x.shape[0], 1)
        or y.shape != (y.shape[0], 1)
        or theta.shape != (2, 1)
    ):
        return None
    if not (
        isinstance(x, np.ndarray)
        and isinstance(y, np.ndarray)
        and isinstance(theta, np.ndarray)
    ):
        return None

    m = x.shape[0]
    X_prime = np.hstack(
        (np.ones((m, 1)), x)
    )  # Add a column of ones to x to perform the vectorized solution
    h_theta = np.dot(X_prime, theta)
    diff = h_theta - y

    gradient = np.dot(X_prime.T, diff) / m

    return gradient


def fit_(x, y, theta, alpha, max_iter):
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None
    if (
        x.shape != (x.shape[0], 1)
        or y.shape != (y.shape[0], 1)
        or theta.shape != (2, 1)
    ):
        return None
    if not (
        isinstance(x, np.ndarray)
        and isinstance(y, np.ndarray)
        and isinstance(theta, np.ndarray)
    ):
        return None

    new_theta = theta.astype(float).copy()
    for _ in range(max_iter):
        gradient = simple_gradient(x, y, new_theta)
        new_theta -= alpha * gradient

    return new_theta


def predict(x, theta):
    return np.dot(np.hstack((np.ones((x.shape[0], 1)), x)), theta)


if __name__ == "__main__":
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    theta = np.array([1, 1]).reshape((-1, 1))

    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    print(theta1)  # Output: array([[1.40709365], [1.1150909 ]])

    predictions = predict(x, theta1)
    print(
        predictions
    )  # Output: array([[15.3408728 ], [25.38243697], [36.59126492], [55.95130097], [65.53471499]])
