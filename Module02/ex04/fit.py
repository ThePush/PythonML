import numpy as np


def gradient(x, y, theta):
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None

    if x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]:
        return None

    if not (
        isinstance(x, np.ndarray)
        and isinstance(y, np.ndarray)
        and isinstance(theta, np.ndarray)
    ):
        return None

    m = x.shape[0]
    X_prime = np.hstack((np.ones((m, 1)), x))
    grad = (1 / m) * X_prime.T.dot(X_prime.dot(theta) - y)

    return grad


def fit_(x, y, theta, alpha, max_iter):
    if not (
        isinstance(x, np.ndarray)
        and isinstance(y, np.ndarray)
        and isinstance(theta, np.ndarray)
    ):
        return None

    if x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]:
        return None

    new_theta = theta.copy()
    for _ in range(max_iter):
        grad = gradient(x, y, new_theta)
        new_theta -= alpha * grad

    return new_theta


def predict_(x, theta):
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None

    X_prime = np.hstack((np.ones((x.shape[0], 1)), x))
    return X_prime.dot(theta)


if __name__ == "__main__":
    x = np.array(
        [[0.2, 2.0, 20.0], [0.4, 4.0, 40.0], [0.6, 6.0, 60.0], [0.8, 8.0, 80.0]]
    )
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta = np.array([[42.0], [1.0], [1.0], [1.0]])
    theta2 = fit_(x, y, theta, alpha=0.0005, max_iter=42000)
    print(theta2)
    # Output: array([[41.99..],[0.97..], [0.77..], [-1.20..]])
    print(predict_(x, theta2))
    # Output: array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])
