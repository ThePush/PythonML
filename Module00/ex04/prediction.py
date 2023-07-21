import numpy as np


def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.array x.
    Args:
        x: has to be a numpy.array of dimension m * n.
    Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None
    return np.c_[np.ones(x.shape[0]), x]


def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x and/or theta are not numpy.array.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None
    if not isinstance(theta, np.ndarray) or theta.size == 0:
        return None
    if len(x.shape) != 1:
        return None
    if theta.shape[0] != 2 or theta.shape[1] != 1:
        return None
    x = add_intercept(x)
    return np.dot(x, theta)


if __name__ == "__main__":
    x = np.arange(1, 6)
    # Example 1:
    theta1 = np.array([[5], [0]])
    print(
        f"Test: x=np.arange(1,6)\ntheta1 = np.array([[5], [0]])\nExpected Ouput:\n [[5.]\n [5.]\n [5.]\n [5.]\n [5.]]\nActual Output:\n {predict_(x, theta1)}\n"
    )
    # Do you remember why y_hat contains only 5’s here?

    # Example 2:
    theta2 = np.array([[0], [1]])
    print(
        f"Test: x=np.arange(1,6)\ntheta2 = np.array([[0], [1]])\nExpected Ouput:\n [[1.]\n [2.]\n [3.]\n [4.]\n [5.]]\nActual Output:\n {predict_(x, theta2)}\n"
    )
    # Do you remember why y_hat == x here?

    # Example 3:
    theta3 = np.array([[5], [3]])
    print(
        f"Test: x=np.arange(1,6)\ntheta3 = np.array([[5], [3]])\nExpected Ouput:\n [[ 8.]\n [11.]\n [14.]\n [17.]\n [20.]]\nActual Output:\n {predict_(x, theta3)}\n"
    )

    # Example 4:
    theta4 = np.array([[-3], [1]])
    print(
        f"Test: x=np.arange(1,6)\ntheta4 = np.array([[-3], [1]])\nExpected Ouput:\n [[-2.]\n [-1.]\n [ 0.]\n [ 1.]\n [ 2.]]\nActual Output:\n {predict_(x, theta4)}\n"
    )
