import numpy as np


def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
    x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
    The sigmoid value as a numpy.ndarray of shape (m, 1).
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if x.size == 0:
        return None

    return 1 / (1 + np.exp(-x))


class MyLogisticRegression:
    """
    Description:
    My personal logistic regression to classify things.
    """

    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta

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
            self.theta -= self.alpha * gradient

        return self.theta


if __name__ == "__main__":
    np.set_printoptions(precision=8)  # Set the precision for printing

    X = np.array([[1.0, 1.0, 2.0, 3.0], [5.0, 8.0, 13.0, 21.0], [3.0, 5.0, 9.0, 14.0]])
    Y = np.array([[1], [0], [1]])
    thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
    mylr = MyLogisticRegression(thetas)

    # Example 0:
    print(mylr.predict_(X))
    # Output:
    # array([[0.99930437],
    #        [1.        ],
    #        [1.        ]])

    # Example 1:
    print(mylr.loss_(Y, mylr.predict_(X)))
    # Output: 11.51315742

    # Example 2:
    mylr.fit_(X, Y)
    print(mylr.theta)
    # Output:
    # array([[ 2.11826435],
    #        [ 0.10154334],
    #        [ 6.43942899],
    #        [-5.10817488],
    #        [ 0.6212541 ]])

    # Example 3:
    print(mylr.predict_(X))
    # Output:
    # array([[0.57606717],
    #        [0.68599807],
    #        [0.06562156]])

    # Example 4:
    print(mylr.loss_(Y, mylr.predict_(X)))
    # Output: 1.47791269
