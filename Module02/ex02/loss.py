import numpy as np


def loss_(y, y_hat):
    """Computes the mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.

    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.

    Return:
    The mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.array.
    None if y and y_hat does not share the same dimensions.
    None if y or y_hat is not of expected type.

    Raises:
    This function should not raise any Exception."""

    if not isinstance(y, np.ndarray) or y.size == 0:
        return None
    if not isinstance(y_hat, np.ndarray) or y_hat.size == 0:
        return None
    if y.shape != y_hat.shape:
        return None

    m = y.shape[0]
    loss = np.dot((y_hat - y).T, (y_hat - y)) / (2 * m)

    return loss.item()


if __name__ == "__main__":
    # Example usage:
    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))

    print(loss_(X, Y))  # Output: 2.142857142857143
    print(loss_(X, X))  # Output: 0.0
