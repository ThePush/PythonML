import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR


def plot_predictions_univariate(
    x, y, y_pred, title, xlabel="Feature", ylabel="y: sell price (in keuros)"
):
    plt.scatter(x, y, label="Sell price", zorder=3)
    plt.scatter(x, y_pred, label="Predicted sell price", zorder=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, zorder=0)
    plt.show()


def plot_predictions_multivariate(
    x, y, y_pred, title, xlabel, ylabel="y: sell price (in keuros)"
):
    plt.scatter(x, y, label="Actual", zorder=3)
    plt.scatter(x, y_pred, label="Predicted", zorder=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, zorder=1)
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("spacecraft_data.csv")

    # Age
    X_age = np.array(data[["Age"]])
    Y = np.array(data[["Sell_price"]])
    myLR_age = MyLR(thetas=np.array([[923.0], [-35.0]]), alpha=2.5e-5, max_iter=100000)
    myLR_age.fit_(X_age, Y)
    y_pred_age = myLR_age.predict_(X_age)
    if y_pred_age is not None:
        plot_predictions_univariate(
            X_age,
            Y,
            y_pred_age,
            "Univariate - Age vs Sell_price",
            "$x_{1}$: age (in years)",
        )
    print("MSE for Age model:", myLR_age.mse_(y_pred_age, Y))

    # Thrust_power
    X_thrust = np.array(data[["Thrust_power"]])
    myLR_thrust = MyLR(thetas=np.array([[240.0], [3.0]]), alpha=2.5e-5, max_iter=100000)
    myLR_thrust.fit_(X_thrust, Y)
    y_pred_thrust = myLR_thrust.predict_(X_thrust)
    if y_pred_thrust is not None:
        plot_predictions_univariate(
            X_thrust,
            Y,
            y_pred_thrust,
            "Univariate - Thrust_power vs Sell_price",
            "$x_{2}$: thrust power (in 10Km/s)",
        )
    print("MSE for Thrust_power model:", myLR_thrust.mse_(y_pred_thrust, Y))

    # Terameters
    X_distance = np.array(data[["Terameters"]])
    myLR_distance = MyLR(
        thetas=np.array([[900.0], [-4.0]]), alpha=2.5e-5, max_iter=100000
    )
    myLR_distance.fit_(X_distance, Y)
    y_pred_distance = myLR_distance.predict_(X_distance)
    if y_pred_distance is not None:
        plot_predictions_univariate(
            X_distance,
            Y,
            y_pred_distance,
            "Univariate - Terameters vs Sell_price",
            "$x_{3}$: distance totalizer value of spacecraft (in Tmeters)",
        )
    print("MSE for Terameters model:", myLR_distance.mse_(y_pred_distance, Y))

    X = np.array(data[["Age", "Thrust_power", "Terameters"]])
    Y = np.array(data[["Sell_price"]])

    my_lreg = MyLR(thetas=np.ones((4, 1), dtype=float), alpha=5e-5, max_iter=300000)
    my_lreg.fit_(X, Y)
    y_pred = my_lreg.predict_(X)

    print("Theta:", my_lreg.thetas)
    print("MSE:", my_lreg.mse_(y_pred, Y))

    plot_predictions_multivariate(
        data["Age"],
        Y,
        y_pred,
        "Multivariate - Age vs Sell_price",
        "$x_{1}$: age (in years)",
    )
    plot_predictions_multivariate(
        data["Thrust_power"],
        Y,
        y_pred,
        "Multivariate - Thrust_power vs Sell_price",
        "$x_{2}$: thrust power (in 10Km/s)",
    )
    plot_predictions_multivariate(
        data["Terameters"],
        Y,
        y_pred,
        "Multivariate - Terameters vs Sell_price",
        "$x_{3}$: distance totalizer value of spacecraft (in Tmeters)",
    )
