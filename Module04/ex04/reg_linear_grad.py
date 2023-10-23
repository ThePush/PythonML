import numpy as np


def reg_linear_grad(y, x, theta, lambda_):
    if (
        y.size == 0
        or x.size == 0
        or theta.size == 0
        or y.shape[0] != x.shape[0]
        or x.shape[1] + 1 != theta.shape[0]
    ):
        return None

    m, n = x.shape
    x_prime = np.column_stack((np.ones((m, 1)), x))
    theta_prime = theta.copy()
    theta_prime[0] = 0
    gradient = np.zeros((n + 1, 1))

    for j in range(n + 1):
        for i in range(m):
            gradient[j] += (np.dot(x_prime[i], theta) - y[i]) * x_prime[i, j]
        if j != 0:
            gradient[j] += lambda_ * theta[j]

    gradient /= m

    return gradient


def vec_reg_linear_grad(y, x, theta, lambda_):
    if (
        y.size == 0
        or x.size == 0
        or theta.size == 0
        or y.shape[0] != x.shape[0]
        or x.shape[1] + 1 != theta.shape[0]
    ):
        return None

    m, n = x.shape
    x_prime = np.column_stack((np.ones((m, 1)), x))
    theta_prime = theta.copy()
    theta_prime[0] = 0

    gradient = (1 / m) * (
        np.dot(x_prime.T, (np.dot(x_prime, theta) - y)) + lambda_ * theta_prime
    )

    return gradient


if __name__ == "__main__":
    x = np.array(
        [
            [-6, -7, -9],
            [13, -2, 14],
            [-7, 14, -1],
            [-8, -4, 6],
            [-5, -9, 6],
            [1, -5, 11],
            [9, -11, 8],
        ]
    )
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    theta = np.array([[7.01], [3], [10.5], [-6]])

    # Example 1.1:
    print(reg_linear_grad(y, x, theta, 1))
    # Output:
    # array([[ -60.99      ],
    #        [-195.64714286],
    #        [ 863.46571429],
    #        [-644.52142857]])

    # Example 1.2:
    print(vec_reg_linear_grad(y, x, theta, 1))
    # Output:
    # array([[ -60.99      ],
    #        [-195.64714286],
    #        [ 863.46571429],
    #        [-644.52142857]])

    # Example 2.1:
    print(reg_linear_grad(y, x, theta, 0.5))
    # Output:
    # array([[ -60.99      ],
    #        [-195.86142857],
    #        [ 862.71571429],
    #        [-644.09285714]])

    # Example 2.2:
    print(vec_reg_linear_grad(y, x, theta, 0.5))
    # Output:
    # array([[ -60.99      ],
    #        [-195.86142857],
    #        [ 862.71571429],
    #        [-644.09285714]])

    # Example 3.1:
    print(reg_linear_grad(y, x, theta, 0.0))
    # Output:
    # array([[ -60.99      ],
    #        [-196.07571429],
    #        [ 861.96571429],
    #        [-643.66428571]])

    # Example 3.2:
    print(vec_reg_linear_grad(y, x, theta, 0.0))
    # Output:
    # array([[ -60.99      ],
    #        [-196.07571429],
    #        [ 861.96571429],
    #        [-643.66428571]])
