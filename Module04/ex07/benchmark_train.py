import pickle
import numpy as np
import pandas as pd
from ridge import MyRidge
from data_spliter import data_spliter
from polynomial_model import add_polynomial_features
from z_score import zscore
from matplotlib import pyplot as plt
import sys
import os
import copy

import numpy as np

if __name__ == "__main__":
    if not os.path.exists("space_avocado.csv"):
        print(f"file not found")
        sys.exit(1)

    # Load data
    data = pd.read_csv("space_avocado.csv")
    data.drop("Unnamed: 0", axis=1, inplace=True)
    x = data[["weight", "prod_distance", "time_delivery"]].to_numpy().reshape(-1, 3)
    y = data["target"].to_numpy().reshape(-1, 1)
    x_train, _, y_train, _ = data_spliter(x, y, 0.8)

    # Train models
    models = []
    for degree in range(1, 5):
        for lambda_ in np.arange(0.0, 1.2, step=0.2):
            print(
                f"Training model with polynomial degree: {degree}, lambda: {lambda_:.1f}"
            )
            model = MyRidge(
                thetas=np.ones(shape=(3 * degree + 1, 1)),
                alpha=0.005,
                max_iter=100_000,
                lambda_=lambda_,
                degree=degree,
            )
            x_poly_norm = zscore(add_polynomial_features(x_train, degree))
            model.set_params(thetas=np.ones(shape=(3 * degree + 1, 1)))
            model.fit_(x_poly_norm, y_train)
            y_hat = model.predict_(x_poly_norm)
            model.loss = model.loss_(y_train, y_hat)
            models.append(copy.deepcopy(model))
            print(f"Loss: {model.loss:.1f}")

    # Save thetas
    with open("models.pickle", "wb") as f:
        pickle.dump(models, f)

    # Plot training losses
    plt.title("Loss vs Degree and Lambda")
    xticks = [f"Degree {model.degree} λ/{model.lambda_:.1f}" for model in models]
    plt.xticks(np.arange(len(xticks)), xticks, rotation=90)
    plt.plot(np.arange(len(models)), [model.loss for model in models])
    plt.ylabel("Loss")
    plt.xlabel("Degree and Lambda")
    plt.show()
