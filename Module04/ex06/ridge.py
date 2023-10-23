import numpy as np
from mylinearregression import MyLinearRegression


class MyRidge(MyLinearRegression):
    """
    Description: My personal linear regression class to fit like a boss.
    """

    def __init__(
        self,
        thetas,
        alpha=0.001,
        max_iter=1000,
        lambda_=0.5,
        degree=4,
    ):
        super().__init__(thetas, alpha, max_iter)
        self.lambda_ = lambda_
        self.degree = degree
        self.loss = None

    def get_params(self):
        return vars(self)

    def set_params(self, **params):
        for key, value in params.items():
            if key in vars(self).keys():
                setattr(self, key, value)
        return self

    def l2(self, theta):
        theta_prime = theta.copy()
        theta_prime[0] = 0
        l2_reg = np.dot(theta_prime.T, theta_prime)
        return l2_reg[0][0]

    def gradient(self, x, y, theta):
        m, n = x.shape
        x_prime = np.column_stack((np.ones((m, 1)), x))
        theta_prime = theta.copy()
        theta_prime[0] = 0
        grad = (1 / m) * (
            np.dot(x_prime.T, (np.dot(x_prime, theta) - y)) + self.lambda_ * theta_prime
        )
        return grad

    def fit_(self, x, y):
        new_theta = self.thetas.copy()
        for _ in range(self.max_iter):
            grad = self.gradient(x, y, new_theta)
            grad_clipped = np.clip(grad, -1e8, 1e8)
            new_theta -= self.alpha * grad_clipped
        self.thetas = new_theta
        return self.thetas

    def loss_(self, y, y_hat):
        m = y.shape[0]
        loss = (1 / (2 * m)) * (
            np.dot((y_hat - y).T, (y_hat - y)) + self.lambda_ * self.l2(self.thetas)
        )
        return loss[0][0]

    def mse_(self, y, y_hat):
        if y.shape != y_hat.shape:
            return None
        return np.mean((y_hat - y) ** 2) + (self.lambda_ / (2 * y.shape[0])) * self.l2(
            self.thetas
        )

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        y_hat = self.predict_(x)
        u = np.square(y - y_hat).sum()
        v = np.square(y - y).sum()
        return 1 - u / v
