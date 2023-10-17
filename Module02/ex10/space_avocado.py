import pandas as pd
import os
from data_spliter import data_spliter
from mylinearregression import MyLinearRegression as MyLR
from polynomial_model import add_polynomial_features
import pickle
from matplotlib import pyplot as plt
from z_score import zscore
import sys


if __name__ == "__main__":
    if not os.path.exists("models.pickle") or not os.path.exists("space_avocado.csv"):
        print(f"file not found")
        sys.exit(1)

    # Load data
    data = pd.read_csv("space_avocado.csv")
    X = data[["weight", "prod_distance", "time_delivery"]].to_numpy().reshape(-1, 3)
    Y = data["target"].to_numpy().reshape(-1, 1)
    x_train, x_test, y_train, y_test = data_spliter(X, Y, 0.8)

    with open("models.pickle", "rb") as f:
        saved_thetas = pickle.load(f)

    # Predict and save results
    mse = []
    y_preds = []
    for i, thetas in enumerate(saved_thetas):
        degree = i + 1
        model = MyLR(thetas)
        x_test_poly_norm = zscore(add_polynomial_features(x_test, degree))
        y_pred = model.predict_(x_test_poly_norm)
        y_preds.append(y_pred)
        mse.append(model.mse_(y_test, y_pred))
        print(f"Degree {degree}, MSE: {mse[-1]}")

    # Get best model
    best_model = mse.index(min(mse))
    y_pred = y_preds[best_model]
    print(f"Best model: {best_model + 1} degree polynomial")

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        x_test[:, 0], x_test[:, 1], y_test, c="r", marker="o", label="True Price"
    )
    ax.scatter(
        x_test[:, 0], x_test[:, 1], y_pred, c="b", marker="^", label="Predicted Price"
    )
    ax.set_xlabel("Weight")
    ax.set_ylabel("Prod Distance")
    ax.set_zlabel("Price")
    ax.legend()
    plt.show()

    # 2D plots
    for i, feature in enumerate(["weight", "prod_distance", "time_delivery"]):
        plt.title(feature)
        x_axis = x_test[:, i].reshape(-1, 1)
        plt.scatter(x_axis, y_test, label="Actual values")
        plt.scatter(x_axis, y_pred, label="Predictions")
        plt.legend()
        plt.show()
