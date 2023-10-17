import pickle
import numpy as np
import pandas as pd
from mylinearregression import MyLinearRegression as MyLR
from data_spliter import data_spliter
from polynomial_model import add_polynomial_features
from z_score import zscore
from matplotlib import pyplot as plt
import sys
import os


if __name__ == "__main__":
    if not os.path.exists("models.pickle") or not os.path.exists("space_avocado.csv"):
        print(f"file not found")
        sys.exit(1)

    # Load data
    data = pd.read_csv("space_avocado.csv")
    data.drop("Unnamed: 0", axis=1, inplace=True)
    x = data[["weight", "prod_distance", "time_delivery"]].to_numpy().reshape(-1, 3)
    y = data["target"].to_numpy().reshape(-1, 1)
    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)

    # Train models
    saved_thetas = []
    mse = []
    for i in range(4):
        degree = i + 1
        x_test_poly_norm = zscore(add_polynomial_features(x_train, degree))
        model = MyLR(
            np.zeros((x_test_poly_norm.shape[1] + 1, 1)),
            alpha=0.005,
            max_iter=1_000_000,
        )
        print(f"Training Degree {degree} model...")
        model.fit_(x_test_poly_norm, y_train)
        saved_thetas.append(model.thetas)
        mse.append(model.mse_(y_train, model.predict_(x_test_poly_norm)))
        print(f"Degree {degree} MSE: {mse[-1]}")
        print(f"Thetas: {model.thetas}")

    # Save thetas
    with open("models.pickle", "wb") as f:
        pickle.dump(saved_thetas, f)

    # Plot training losses
    plt.title("MSE vs Degree")
    plt.bar(range(1, len(mse) + 1), mse)
    plt.xlabel("Degree")
    plt.ylabel("MSE score")
    plt.show()
