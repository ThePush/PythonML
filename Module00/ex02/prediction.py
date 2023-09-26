import numpy as np


def simple_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
    if x.size == 0 or theta.size == 0 or x.ndim != 1 or theta.ndim != 1:
        return None
    return np.array([float(theta[0] + theta[1] * x_i) for x_i in x])


if __name__ == "__main__":
    x = np.arange(1, 6)

    # Example 1:
    theta1 = np.array([5, 0])
    print(
        f"Test: theta1=np.array([5, 0])\nExpected Ouput: array([5., 5., 5., 5., 5.])\nActual Output: {simple_predict(x, theta1)}\n"
    )
    # Do you understand why y_hat contains only 5’s here?
    # 0x+5 = 5

    # Example 2:
    theta2 = np.array([0, 1])
    print(
        f"Test: theta2=np.array([0, 1])\nExpected Ouput: array([1., 2., 3., 4., 5.])\nActual Output: {simple_predict(x, theta2)}\n"
    )
    # Do you understand why y_hat == x here?
    # 1x+0 = x

    # Example 3:
    theta3 = np.array([5, 3])
    print(
        f"Test: theta3=np.array([5, 3])\nExpected Ouput: array([8., 11., 14., 17., 20.])\nActual Output: {simple_predict(x, theta3)}\n"
    )

    # Example 4:
    theta4 = np.array([-3, 1])
    print(
        f"Test: theta4=np.array([-3, 1])\nExpected Ouput: array([-2., -1., 0., 1., 2.])\nActual Output: {simple_predict(x, theta4)}\n"
    )
