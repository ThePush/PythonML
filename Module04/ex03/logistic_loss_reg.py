import numpy as np


def l2(theta):
    if theta.size == 0:
        return None

    theta_prime = theta.copy()
    theta_prime[0] = 0
    l2_reg = np.dot(theta_prime.T, theta_prime)

    return l2_reg[0][0]


def reg_log_loss_(y, y_hat, theta, lambda_):
    if y.size == 0 or y_hat.size == 0 or theta.size == 0 or y.shape != y_hat.shape:
        return None

    m = y.shape[0]
    ones = np.ones(y.shape)
    loss = (-1 / m) * (
        np.dot(y.T, np.log(y_hat)) + np.dot((ones - y).T, np.log(ones - y_hat))
    ) + (lambda_ / (2 * m)) * l2(theta)

    return loss[0][0]


if __name__ == "__main__":
    y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
    y_hat = np.array([0.9, 0.79, 0.12, 0.04, 0.89, 0.93, 0.01]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))

    # Example 1:
    print(reg_log_loss_(y, y_hat, theta, 0.5))
    # Output: 0.43377043716475955

    # Example 2:
    print(reg_log_loss_(y, y_hat, theta, 0.05))
    # Output: 0.13452043716475953

    # Example 3:
    print(reg_log_loss_(y, y_hat, theta, 0.9))
    # Output: 0.6997704371647596
