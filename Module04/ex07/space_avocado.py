import pandas as pd
import numpy as np
import os
from data_spliter import data_spliter
import pickle
from matplotlib import pyplot as plt
from z_score import *
from polynomial_model import add_polynomial_features
import sys
import matplotlib.cm as cm
from ridge import MyRidge


if __name__ == "__main__":
    if not os.path.exists("models.pickle") or not os.path.exists("space_avocado.csv"):
        print(f"file not found")
        sys.exit(1)

    # Load data
    data = pd.read_csv("space_avocado.csv")
    X = data[["weight", "prod_distance", "time_delivery"]].to_numpy().reshape(-1, 3)
    Y = data["target"].to_numpy().reshape(-1, 1)
    x_train, x_temp, y_train, y_temp = data_spliter(X, Y, 0.8)
    x_eval, x_cross_test, y_eval, y_cross_test = data_spliter(
        x_temp, y_temp, 0.5
    )

    with open("models.pickle", "rb") as f:
        models = pickle.load(f)

    best_model = min(models, key=lambda model: model.loss)
    models.pop(models.index(best_model))

    # Fit best model
    print(
        f"Training best model with degree {best_model.degree} and λ={best_model.lambda_}..."
    )
    x_train_poly_norm = zscore(add_polynomial_features(x_train, 4))
    best_model = MyRidge(
        thetas=np.ones(shape=(3 * 4 + 1, 1)),
        alpha=0.005,
        max_iter=100_000,
        lambda_=0.0,
        degree=4,
    )
    best_model.fit_(x_train_poly_norm, y_train)

    models.append(best_model)

    # Predict and save results
    y_preds = []
    for model in models:
        x_train_poly = add_polynomial_features(x_train, model.degree)
        x_cross_test_poly = add_polynomial_features(x_cross_test, model.degree)
        mean, std = x_train_poly.mean(axis=0), x_train_poly.std(axis=0)
        x_cross_test_poly_norm = zscore_(x_cross_test_poly, mean, std)
        y_pred = model.predict_(x_cross_test_poly_norm)
        y_preds.append(y_pred)
        model.loss = model.loss_(y_cross_test, y_pred)
        print(
            f"Degree {model.degree} with lambda={model.lambda_:.1f}, Loss: {model.loss}"
        )

    # Plot training losses
    plt.title("Loss vs Degree and Lambda")
    xticks = [f"Degree {model.degree} λ/{model.lambda_:.1f}" for model in models]
    plt.xticks(np.arange(len(xticks)), xticks, rotation=90)
    plt.plot(np.arange(len(models)), [model.loss for model in models])
    plt.ylabel("Loss")
    plt.xlabel("Degree and Lambda")
    plt.show()

    ## 3D plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # x_train_poly = add_polynomial_features(x_train, model.degree)
    # x_eval_poly = add_polynomial_features(x_eval, model.degree)
    # mean, std = x_train_poly.mean(axis=0), x_train_poly.std(axis=0)
    # x_eval_poly_norm = zscore_(x_eval_poly, mean, std)
    # y_pred = model.predict_(x_eval_poly_norm)
    # ax.scatter(
    #    x_eval_poly_norm[:, 0],
    #    x_eval_poly_norm[:, 1],
    #    y_eval,
    #    c="r",
    #    marker="o",
    #    label="True Price",
    # )
    # ax.scatter(
    #    x_eval_poly_norm[:, 0],
    #    x_eval_poly_norm[:, 1],
    #    y_pred,
    #    c="b",
    #    marker="^",
    #    label="Predicted Price",
    # )
    # ax.set_xlabel("Weight")
    # ax.set_ylabel("Prod Distance")
    # ax.set_zlabel("Price")
    # ax.legend()
    # plt.show()

    # Filter the models with the same degree as the best model
    filtered_models = [model for model in models if model.degree == best_model.degree]

    # 2D plots
    for i, feature in enumerate(["weight", "prod_distance", "time_delivery"]):
        plt.title(feature)
        x_axis = x_eval[:, i].reshape(-1, 1)
        plt.scatter(x_axis, y_eval, label="Actual values", color="blue", s=75)

        # Generate colors for different λ values using a colormap
        colormap = cm.get_cmap("plasma", len(filtered_models))

        # Plot the predictions of the filtered models along with the true prices
        for idx, model in enumerate(filtered_models):
            x_train_poly = add_polynomial_features(x_train, model.degree)
            x_eval_poly = add_polynomial_features(x_eval, model.degree)
            mean, std = x_train_poly.mean(axis=0), x_train_poly.std(axis=0)
            x_eval_poly_norm = zscore_(x_eval_poly, mean, std)
            y_pred = model.predict_(x_eval_poly_norm)
            plt.scatter(
                x_axis,
                y_pred,
                label=f"Degree {model.degree}, λ={model.lambda_:.1f}",
                color=colormap(idx),
                alpha=0.5,
            )

        plt.legend()
        plt.show()
