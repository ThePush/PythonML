import numpy as np


def l2(theta):
    if theta.size == 0:
        return None

    theta_prime = theta.copy()
    theta_prime[0] = 0
    l2_reg = np.dot(theta_prime.T, theta_prime)

    return l2_reg[0][0]


def reg_loss_(y, y_hat, theta, lambda_):
    if y.size == 0 or y_hat.size == 0 or theta.size == 0 or y.shape != y_hat.shape:
        return None

    m = y.shape[0]
    loss = (1 / (2 * m)) * (np.dot((y_hat - y).T, (y_hat - y)) + lambda_ * l2(theta))

    return loss[0][0]


if __name__ == "__main__":
    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))

    # Example 1:
    print(reg_loss_(y, y_hat, theta, 0.5))
    # Output: 0.8503571428571429

    # Example 2:
    print(reg_loss_(y, y_hat, theta, 0.05))
    # Output: 0.5511071428571429

    # Example 3:
    print(reg_loss_(y, y_hat, theta, 0.9))
    # Output: 1.116357142857143
