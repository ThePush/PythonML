import numpy as np

class MyLinearRegression:
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

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

if __name__ == "__main__":
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
    Y = np.array([[23.], [48.], [218.]])
    mylr = MyLinearRegression(np.array([[1.], [1.], [1.], [1.], [1.]]))

    # Example 0:
    y_hat = mylr.predict_(X)
    print(y_hat)
    # Output:
    # array([[  8.],
    #        [ 48.],
    #        [323.]])

    # Example 1:
    print(mylr.loss_elem_(Y, y_hat))
    # Output:
    # array([[  225.],
    #        [    0.],
    #        [11025.]])

    # Example 2:
    print(mylr.loss_(Y, y_hat))
    # Output:
    # 1875.0

    # Example 3:
    mylr.alpha = 1.6e-4
    mylr.max_iter = 200000
    mylr.thetas = np.array([[1.], [1.], [1.], [1.], [1.]])
    mylr.thetas = mylr.fit_(X, Y)
    print(mylr.thetas)
    # Output:
    # array([[18.188..],
    #        [ 2.767..],
    #        [-0.374..],
    #        [ 1.392..],
    #        [ 0.017..]])

    # Example 4:
    y_hat = mylr.predict_(X)
    print(y_hat)
    # Output:
    # array([[ 23.417..],
    #        [ 47.489..],
    #        [218.065...]])

    # Example 5:
    print(mylr.loss_elem_(Y, y_hat))
    # Output:
    # array([[0.174..],
    #        [0.260..],
    #        [0.004..]])

    # Example 6:
    print(mylr.loss_(Y, y_hat))
    # Output:
    # 0.0732..