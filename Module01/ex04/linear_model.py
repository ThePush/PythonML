import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MyLinearRegression:
    """
    Description:
    My personal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    @staticmethod
    def simple_gradient(x, y, theta):
        """Computes a gradient vector from three non-empty numpy.array, without any for loop.
        The three arrays must have compatible shapes.

        Args:
        x: has to be a numpy.array, a matrix of shape m * 1.
        y: has to be a numpy.array, a vector of shape m * 1.
        theta: has to be a numpy.array, a 2 * 1 vector.

        Return:
        The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
        None if x, y, or theta is an empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.

        Raises:
        This function should not raise any Exception."""

        if x.size == 0 or y.size == 0 or theta.size == 0:
            return None
        if (
            x.shape != (x.shape[0], 1)
            or y.shape != (y.shape[0], 1)
            or theta.shape != (2, 1)
        ):
            return None
        if not (
            isinstance(x, np.ndarray)
            and isinstance(y, np.ndarray)
            and isinstance(theta, np.ndarray)
        ):
            return None

        m = x.shape[0]
        X_prime = np.hstack(
            (np.ones((m, 1)), x)
        )  # Add a column of ones to x to perform the vectorized solution
        h_theta = np.dot(X_prime, theta)
        diff = h_theta - y

        gradient = np.dot(X_prime.T, diff) / m

        return gradient

    def fit_(self, x, y):
        if x.size == 0 or y.size == 0 or self.thetas.size == 0:
            return None
        if (
            x.shape != (x.shape[0], 1)
            or y.shape != (y.shape[0], 1)
            or self.thetas.shape != (2, 1)
        ):
            return None
        if not (
            isinstance(x, np.ndarray)
            and isinstance(y, np.ndarray)
            and isinstance(self.thetas, np.ndarray)
        ):
            return None

        self.thetas = self.thetas.astype(float).copy()
        for _ in range(self.max_iter):
            gradient = self.simple_gradient(x, y, self.thetas)
            self.thetas -= self.alpha * gradient

        return self.thetas

    def predict_(self, x):
        return np.dot(np.hstack((np.ones((x.shape[0], 1)), x)), self.thetas)

    def loss_elem_(self, y, y_hat):
        """
        Description:
            Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            J_elem: numpy.array, a vector of dimension (number of the training examples,1).
            None if there is a dimension matching problem between X, Y or theta.
            None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray) or y.size == 0:
            return None
        if not isinstance(y_hat, np.ndarray) or y_hat.size == 0:
            return None
        if y.shape != y_hat.shape:
            return None
        return (y_hat - y) ** 2

    def loss_(self, y, y_hat):
        """
        Description:
            Calculates the value of loss function.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            J_value : has to be a float.
            None if there is a dimension matching problem between X, Y or theta.
            None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray) or y.size == 0:
            return None
        if not isinstance(y_hat, np.ndarray) or y_hat.size == 0:
            return None
        if y.shape != y_hat.shape:
            return None
        return np.sum(self.loss_elem_(y, y_hat)) / (2 * y.shape[0])

    def mse_(y, y_hat):
        if y.shape != y_hat.shape:
            return None
        return np.mean((y_hat - y) ** 2)


if __name__ == "__main__":
    data = pd.read_csv("are_blue_pills_magics.csv")
    Xpill = np.array(data["Micrograms"]).reshape(-1, 1)
    Yscore = np.array(data["Score"]).reshape(-1, 1)

    lr1 = MyLinearRegression(np.array([[89.0], [-8]]))
    lr2 = MyLinearRegression(np.array([[89.0], [-6]]))
    Y_model1 = lr1.predict_(Xpill)
    Y_model2 = lr2.predict_(Xpill)

    print("MSE Model 1:", MyLinearRegression.mse_(Yscore, Y_model1))
    print("MSE Model 2:", MyLinearRegression.mse_(Yscore, Y_model2))

    # Data and hypothesis plot
    plt.grid(True)
    plt.scatter(Xpill, Yscore, label="Strue(pills)", color="cyan")
    plt.plot(
        Xpill,
        Y_model1,
        label="Spredict(pills)_1",
        color="orchid",
        linestyle="--",
        marker="x",
    )  # Add marker='x'
    plt.plot(
        Xpill,
        Y_model2,
        label="Spredict(pills)_2",
        color="lawngreen",
        linestyle="--",
        marker="x",
    )  # Add marker='x'

    plt.xlabel("Quantity of blue pill (in micrograms)")
    plt.ylabel("Space driving score")
    plt.legend()
    plt.title("Spacecraft Piloting Score vs. Blue Pills")
    plt.show()

    # Loss function plot for different theta0 values
    theta1_range = np.linspace(-14, -4)
    theta0_values = [80, 85, 90, 95, 100]

    for theta0 in theta0_values:
        loss_values = [
            lr1.loss_(
                Yscore,
                MyLinearRegression(np.array([[theta0], [theta1]])).predict_(Xpill),
            )
            for theta1 in theta1_range
        ]
        plt.plot(theta1_range, loss_values, label=f"Theta0 = {theta0}")

    plt.grid(True)
    plt.xlabel("θ1")
    plt.ylabel("cost function J(θ0, θ1)")
    plt.legend()
    plt.title("Loss Function J(θ) for Different Theta0 Values")
    plt.show()
