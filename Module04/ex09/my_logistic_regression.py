import numpy as np


def sigmoid_(x):
    if x.size == 0:
        return None
    return 1 / (1 + np.exp(-x))


class MyLogisticRegression:
    """
    Description:
    My personal logistic regression to classify things.
    """

    supported_penalities = [
        "l2"
    ]  # We consider l2 penalty only. One may want to implement other penalties

    def __init__(self, theta, alpha=0.001, max_iter=1000, penalty="l2", lambda_=1.0):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta
        self.penalty = penalty
        self.lambda_ = lambda_ if penalty in self.supported_penalities else 0

    def predict_(self, x):
        if not isinstance(x, np.ndarray) or x.size == 0:
            return None
        ones = np.ones(shape=(x.shape[0], 1))
        x_prime = np.hstack((ones, x))
        return sigmoid_(x_prime.dot(self.theta))

    def vec_log_loss_(self, y, y_hat, eps=1e-15):
        if y.size == 0 or y_hat.size == 0:
            return None

        m = y.shape[0]
        ones = np.ones((m, 1))
        loss = -(1 / m) * np.sum(
            y * np.log(y_hat + eps) + (ones - y) * np.log(ones - y_hat + eps)
        )

        return loss

    def loss_(self, y, y_hat):
        return self.vec_log_loss_(y, y_hat)

    def fit_(self, x, y):
        if x.size == 0 or y.size == 0 or self.theta.size == 0:
            return None

        if x.shape[1] + 1 != self.theta.shape[0]:
            return None

        m = x.shape[0]
        x_prime = np.hstack((np.ones((m, 1)), x))

        for _ in range(self.max_iter):
            y_hat = self.predict_(x)
            gradient = (1 / m) * x_prime.T.dot(y_hat - y)

            if self.penalty == "l2":
                gradient[1:] += (self.lambda_ / m) * self.theta[1:]

            self.theta -= self.alpha * gradient

        return self.theta


if __name__ == "__main__":
    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

    # Example 1:
    model1 = MyLogisticRegression(theta, lambda_=5.0)
    print(model1.penalty)  # Output: 'l2'
    print(model1.lambda_)  # Output: 5.0

    # Example 2:
    model2 = MyLogisticRegression(theta, penalty=None)
    print(model2.penalty)  # Output: None
    print(model2.lambda_)  # Output: 0.0

    # Example 3:
    model3 = MyLogisticRegression(theta, penalty=None, lambda_=2.0)
    print(model3.penalty)  # Output: None
    print(model3.lambda_)  # Output: 0.0
