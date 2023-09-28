import numpy as np


class MyLinearRegression:
    def __init__(self, thetas, alpha=0.0001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas.astype("float64")

    def gradient(self, x, y, theta):
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

    def fit_(self, x, y):
        if not (
            isinstance(x, np.ndarray)
            and isinstance(y, np.ndarray)
            and isinstance(self.thetas, np.ndarray)
        ):
            return None

        if x.shape[0] != y.shape[0] or x.shape[1] + 1 != self.thetas.shape[0]:
            return None

        new_theta = self.thetas.copy()
        for _ in range(self.max_iter):
            grad = self.gradient(x, y, new_theta)
            new_theta -= self.alpha * grad

        self.thetas = new_theta
        return self.thetas

    def predict_(self, x):
        if not isinstance(x, np.ndarray) or not isinstance(self.thetas, np.ndarray):
            return None

        X_prime = np.hstack((np.ones((x.shape[0], 1)), x))
        return X_prime.dot(self.thetas)

    def loss_elem_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or y.size == 0:
            return None
        if not isinstance(y_hat, np.ndarray) or y_hat.size == 0:
            return None
        if y.shape != y_hat.shape:
            return None
        return (y_hat - y) ** 2

    def loss_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or y.size == 0:
            return None
        if not isinstance(y_hat, np.ndarray) or y_hat.size == 0:
            return None
        if y.shape != y_hat.shape:
            return None
        return np.sum(self.loss_elem_(y, y_hat)) / (2 * y.shape[0])

    def mse_(self, y, y_hat):
        if y.shape != y_hat.shape:
            return None
        return np.mean((y_hat - y) ** 2)
