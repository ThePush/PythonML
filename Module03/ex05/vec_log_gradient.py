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
    if x.size == 0:
        return None

    return 1 / (1 + np.exp(-x))


def logistic_predict_(x, theta):
    """
    Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
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
    if x.size == 0 or theta.size == 0:
        return None

    if x.shape[1] + 1 != theta.shape[0]:
        return None

    x_prime = np.hstack((np.ones((x.shape[0], 1)), x))
    y_hat = sigmoid_(x_prime.dot(theta))

    return y_hat


def vec_log_gradient(x, y, theta):
    """
    Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The three arrays must have compatible dimensions.
    Args:
    x: has to be an numpy.ndarray, a matrix of shape m * n.
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    theta: has to be an numpy.ndarray, a vector (n +1) * 1.
    Returns:
    The gradient as a numpy.ndarray, a vector of shape n * 1, containg the result of the formula for all j.
    None if x, y, or theta are empty numpy.ndarray.
    None if x, y and theta do not have compatible shapes.
    Raises:
    This function should not raise any Exception.
    """
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None

    if x.shape[1] + 1 != theta.shape[0]:
        return None

    m = x.shape[0]
    x_prime = np.hstack((np.ones((m, 1)), x))
    y_hat = logistic_predict_(x, theta)

    gradient = (1 / m) * x_prime.T.dot(y_hat - y)

    return gradient


if __name__ == "__main__":
    np.set_printoptions(precision=8)

    # Example 1:
    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    print(vec_log_gradient(x1, y1, theta1))
    # Output:
    # array([[-0.01798621],
    #        [-0.07194484]])

    # Example 2:
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print(vec_log_gradient(x2, y2, theta2))
    # Output:
    # array([[0.3715235 ],
    #        [3.25647547]])

    # Example 3:
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(vec_log_gradient(x3, y3, theta3))
    # Output:
    # array([[-0.55711039 ],
    #        [-0.90334809 ],
    #        [-2.01756886  ],
    #        [-2.10071291 ],
    #        [-3.27257351 ]])
