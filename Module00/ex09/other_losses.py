import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt


def mse_(y, y_hat):
    """
    Description:
        Calculate the MSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        mse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or y.size == 0:
        return None
    if not isinstance(y_hat, np.ndarray) or y_hat.size == 0:
        return None
    if y.shape != y_hat.shape:
        return None
    return (np.dot((y_hat - y).T, (y_hat - y)) / y.shape[0]).item()


def rmse_(y, y_hat):
    """
    Description:
        Calculate the RMSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        rmse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or y.size == 0:
        return None
    if not isinstance(y_hat, np.ndarray) or y_hat.size == 0:
        return None
    if y.shape != y_hat.shape:
        return None
    mse = mse_(y, y_hat)
    if mse is None:
        return None
    return sqrt(mse)


def mae_(y, y_hat):
    """
    Description:
        Calculate the MAE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        mae: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or y.size == 0:
        return None
    if not isinstance(y_hat, np.ndarray) or y_hat.size == 0:
        return None
    if y.shape != y_hat.shape:
        return None
    return (np.sum(np.absolute(y_hat - y)) / y.shape[0]).item()


def r2score_(y, y_hat):
    """
    Description:
        Calculate the R2score between the predicted output and the output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        r2score: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or y.size == 0:
        return None
    if not isinstance(y_hat, np.ndarray) or y_hat.size == 0:
        return None
    if y.shape != y_hat.shape:
        return None
    return 1 - np.dot((y_hat - y).T, (y_hat - y)) / np.dot(
        (y - np.mean(y)).T, (y - np.mean(y))
    )


if __name__ == "__main__":
    # Example 1:
    x = np.array([0, 15, -9, 7, 12, 3, -21])
    y = np.array([2, 14, -13, 5, 12, 4, -19])

    # Mean squared error
    print(f"MSE Expected Output: 4.285714285714286\n Actual Output: {mse_(x, y)}")
    ## sklearn implementation
    print(f" Sklearn: {mean_squared_error(x, y)}\n")

    # Root mean squared error
    print(f"RMSE Expected Output: 2.0701966780270626\n Actual Output: {rmse_(x, y)}")
    ## sklearn implementation not available: take the square root of MSE
    print(f" Sklearn: {sqrt(mean_squared_error(x, y))}\n")

    # Mean absolute error
    print(f"MAE Expected Output: 1.7142857142857142\n Actual Output: {mae_(x, y)}")
    ## sklearn implementation
    print(f" Sklearn: {mean_absolute_error(x, y)}\n")

    # R2-score
    print(
        f"R2-score Expected Output: 0.9681721733858745\n Actual Output: {r2score_(x, y)}"
    )
    ## sklearn implementation
    print(f" Sklearn: {r2_score(x, y)}\n")
