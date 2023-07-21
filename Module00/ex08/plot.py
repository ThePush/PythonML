import numpy as np
import matplotlib.pyplot as plt


def loss_(y, y_hat):
    """Computes the half mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        The half mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or y.size == 0:
        return None
    if not isinstance(y_hat, np.ndarray) or y_hat.size == 0:
        return None
    if y.shape != y_hat.shape:
        return None
    return np.dot((y_hat - y).T, (y_hat - y)) / y.shape[0]


def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
        You must implement a function which plots the data, the prediction line, and the loss.
    You will plot the x and y coordinates of all data points as well as the prediction line
    generated by your theta parameters. Your function must also display the overall loss (J)
    in the title, and draw small lines marking the distance between each data point and its
    predicted value.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exception.
    """
    y_hat = theta[0] + theta[1] * x
    plt.scatter(x, y, color="blue")
    plt.plot(x, y_hat, color="red")
    plt.title(f"Loss: {loss_(y, y_hat)}")
    for i in range(x.shape[0]):
        plt.plot([x[i], x[i]], [y[i], y_hat[i]], color="red", linestyle="--")
    plt.show()


if __name__ == "__main__":
    x = np.arange(1, 6)
    y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])

    # Example 1:
    theta1 = np.array([18, -1])
    plot_with_loss(x, y, theta1)

    # Example 2:
    theta2 = np.array([14, 0])
    plot_with_loss(x, y, theta2)

    # Example 3:
    theta3 = np.array([12, 0.8])
    plot_with_loss(x, y, theta3)
