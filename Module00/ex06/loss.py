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
        return 1
    if not isinstance(theta, np.ndarray) or theta.size == 0:
        return 2
    if len(x.shape) != 1:
        return 3
    if theta.shape[0] != 2 or theta.shape[1] != 1:
        return 4
    x = add_intercept(x)
    return np.dot(x, theta)


def loss_elem_(y, y_hat):
    """
    Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        J_elem: numpy.array, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or y.size == 0:
        return None
    if not isinstance(y_hat, np.ndarray) or y_hat.size == 0:
        return None
    if y.shape != y_hat.shape:
        return None
    return (y_hat - y) ** 2


def loss_(y, y_hat):
    """
    Description:
        Calculates the value of loss function.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or y.size == 0:
        return None
    if not isinstance(y_hat, np.ndarray) or y_hat.size == 0:
        return None
    if y.shape != y_hat.shape:
        return None
    return np.sum(loss_elem_(y, y_hat)) / (2 * y.shape[0])


if __name__ == "__main__":
    x1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    theta1 = np.array([[2.0], [4.0]])
    y_hat1 = predict_(x1, theta1)
    y1 = np.array([[2.0], [7.0], [12.0], [17.0], [22.0]])

    # Example 1:
    print(f"Test 1")
    print(f"Expected Ouput:\n [[0.]\n [1.]\n [4.]\n [9.]\n [16.]]")
    print(f"Actual Output:\n {loss_elem_(y1, y_hat1)}\n")

    # Example 2:
    print(f"Test 2")
    print(f"Expected Ouput:\n 3.0")
    print(f"Actual Output:\n {loss_(y1, y_hat1)}\n")

    x2 = np.array([0, 15, -9, 7, 12, 3, -21])
    theta2 = np.array([[0.0], [1.0]]).reshape(-1, 1)
    y_hat2 = predict_(x2, theta2)
    y2 = np.array([[2], [14], [-13], [5], [12], [4], [-19]]).reshape(-1, 1)

    # Example 3:
    print(f"Test 3")
    print(f"Expected Ouput:\n 2.142857142857143")
    print(f"Actual Output:\n {loss_(y2, y_hat2)}\n")

    # Example 4:
    print(f"Test 4")
    print(f"Expected Ouput:\n 0.0")
    print(f"Actual Output:\n {loss_(y2, y2)}\n")
