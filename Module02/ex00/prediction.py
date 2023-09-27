import numpy as np


def simple_predict(x, theta):
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None
    if not isinstance(theta, np.ndarray) or theta.size == 0:
        return None
    if x.shape[1] + 1 != theta.shape[0] or theta.shape[1] != 1:
        return None

    m, n = x.shape
    y_hat = np.zeros((m, 1))

    for i in range(m):
        y_hat[i] = theta[0]  # Add theta0 (bias term) to the prediction
        for j in range(n):
            y_hat[i] += (
                x[i, j] * theta[j + 1]
            )  # Add the product of the feature value and corresponding theta

    return y_hat


if __name__ == "__main__":
    # Example usage:
    x = np.arange(1, 13).reshape((4, -1))

    theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
    print(simple_predict(x, theta1))
    # Output: array([[5.], [5.], [5.], [5.]])

    theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
    print(simple_predict(x, theta2))
    # Output: array([[ 1.], [ 4.], [ 7.], [10.]])

    theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
    print(simple_predict(x, theta3))
    # Output: array([[ 9.64], [24.28], [38.92], [53.56]])

    theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
    print(simple_predict(x, theta4))
    # Output: array([[12.5], [32. ], [51.5], [71. ]])
