import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def reg_logistic_grad(y, x, theta, lambda_):
    if y.size == 0 or x.size == 0 or theta.size == 0 or y.shape[0] != x.shape[0]:
        return None

    m, n = x.shape
    x_prime = np.column_stack((np.ones((m, 1)), x))
    theta_prime = theta.copy()
    theta_prime[0] = 0
    gradient = np.zeros((n + 1, 1))

    for j in range(n + 1):
        for i in range(m):
            gradient[j] += (sigmoid(np.dot(x_prime[i], theta)) - y[i]) * x_prime[i, j]
        if j != 0:
            gradient[j] += lambda_ * theta[j]

    gradient /= m

    return gradient


def vec_reg_logistic_grad(y, x, theta, lambda_):
    if y.size == 0 or x.size == 0 or theta.size == 0 or y.shape[0] != x.shape[0]:
        return None

    m, n = x.shape
    x_prime = np.column_stack((np.ones((m, 1)), x))
    theta_prime = theta.copy()
    theta_prime[0] = 0

    gradient = (1 / m) * (
        np.dot(x_prime.T, (sigmoid(np.dot(x_prime, theta)) - y)) + lambda_ * theta_prime
    )

    return gradient


if __name__ == "__main__":
    x = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    y = np.array([[0], [1], [1]])
    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

    # Example 1.1:
    print(reg_logistic_grad(y, x, theta, 1))
    # Output:
    # array([[-0.55711039],
    #        [-1.40334809],
    #        [-1.91756886],
    #        [-2.56737958],
    #        [-3.03924017]])

    # Example 1.2:
    print(vec_reg_logistic_grad(y, x, theta, 1))
    # Output:
    # array([[-0.55711039],
    #        [-1.40334809],
    #        [-1.91756886],
    #        [-2.56737958],
    #        [-3.03924017]])

    # Example 2.1:
    print(reg_logistic_grad(y, x, theta, 0.5))
    # Output:
    # array([[-0.55711039],
    #        [-1.15334809],
    #        [-1.96756886],
    #        [-2.33404624],
    #        [-3.15590684]])

    # Example 2.2:
    print(vec_reg_logistic_grad(y, x, theta, 0.5))
    # Output:
    # array([[-0.55711039],
    #        [-1.15334809],
    #        [-1.96756886],
    #        [-2.33404624],
    #        [-3.15590684]])

    # Example 3.1:
    print(reg_logistic_grad(y, x, theta, 0.0))
    # Output:
    # array([[-0.55711039],
    #        [-0.90334809],
    #        [-2.01756886],
    #        [-2.10071291],
    #        [-3.27257351]])

    # Example 3.2:
    print(vec_reg_logistic_grad(y, x, theta, 0.0))
    # Output:
    # array([[-0.55711039],
    #        [-0.90334809],
    #        [-2.01756886],
    #        [-2.10071291],
    #        [-3.27257351]])
