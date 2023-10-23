import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_logistic_regression import MyLogisticRegression
from data_spliter import data_spliter
from polynomial_model import add_polynomial_features
from z_score import zscore
from other_metrics import f1_score_
import pickle


if __name__ == "__main__":
    data = pd.read_csv("solar_system_census.csv")
    planets = pd.read_csv("solar_system_census_planets.csv")
    X = data.to_numpy()
    y = planets["Origin"].to_numpy().reshape(-1, 1)

    X_poly_norm = zscore(add_polynomial_features(X, 3))
    X_train, x_temp, y_train, y_temp = data_spliter(X_poly_norm, y, proportion=0.8)
    x_cross_test, x_eval, y_cross_test, y_eval = data_spliter(
        x_temp, y_temp, proportion=0.5
    )

    X_train_orig, X_temp_orig, y_train_orig, y_temp_orig = data_spliter(
        X, y, proportion=0.8
    )
    X_cv_orig, X_test_orig, y_cv_orig, y_test_orig = data_spliter(
        X_temp_orig, y_temp_orig, proportion=0.5
    )

    with open("models.pkl", "rb") as f:
        models = pickle.load(f)

    # Find the best model index
    f1_scores = []
    for classifiers in models:
        y_hat = []
        for model in classifiers:
            y_hat.append(model.predict_(x_cross_test))
        y_hat = np.argmax(np.hstack(y_hat), axis=1).reshape(-1, 1)
        f1_scores.append(f1_score_(y_cross_test, y_hat))
    best_model_idx = np.argmax(f1_scores)

    # Train the best model from scratch on the training set
    lambdas = np.linspace(0, 1, 11)
    best_lambda = lambdas[best_model_idx]
    num_classes = len(np.unique(y_train))
    best_classifiers = []
    for c in range(num_classes):
        theta = np.zeros((X_train.shape[1] + 1, 1))
        model = MyLogisticRegression(
            theta, alpha=0.001, max_iter=1000, penalty="l2", lambda_=best_lambda
        )
        y_binary = (y_train == c).astype(int)
        model.fit_(X_train, y_binary)
        best_classifiers.append(model)

    # Calculate F1 scores on the test set
    test_f1_scores = []
    for classifiers in models:
        y_hat = []
        for model in classifiers:
            y_hat.append(model.predict_(x_eval))
        y_hat = np.argmax(np.hstack(y_hat), axis=1).reshape(-1, 1)
        test_f1_scores.append(f1_score_(y_eval, y_hat))

    for i, f1_score in enumerate(test_f1_scores):
        print(f"lambda = {lambdas[i]:.1f}: Test F1 score = {f1_score}")

    y_hat_best = []
    for model in best_classifiers:
        y_hat_best.append(model.predict_(x_eval))
    y_hat_best = np.argmax(np.hstack(y_hat_best), axis=1).reshape(-1, 1)

    plt.bar(lambdas, test_f1_scores, width=0.05)
    plt.xlabel("Lambda")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Lambda")
    plt.xlim(0, 1)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    weight = X_test_orig[:, 0]
    height = X_test_orig[:, 1]
    bone_density = X_test_orig[:, 2]

    zipcodes = [0, 1, 2, 3]
    labels = [
        "The flying cities of Venus (0)",
        "United Nations of Earth (1)",
        "Mars Republic (2)",
        "The Asteroids’ Belt colonies (3)",
    ]
    colors = ["blue", "red", "green", "purple"]

    for zc, label, color in zip(zipcodes, labels, colors):
        mask = (y_test_orig == zc).ravel()
        ax.scatter(
            weight[mask],
            height[mask],
            bone_density[mask],
            c=color,
            marker="o",
            label=f"True {label}",
            s=50,
            alpha=0.5,
        )

    for zc, label, color in zip(zipcodes, labels, colors):
        mask = (y_hat_best == zc).ravel()
        ax.scatter(
            weight[mask],
            height[mask],
            bone_density[mask],
            c=color,
            marker="^",
            label=f"Predicted {label}",
        )

    ax.set_xlabel("Weight")
    ax.set_ylabel("Height")
    ax.set_zlabel("Bone Density")
    ax.set_title("True vs Predicted Zipcodes")
    ax.legend()
    plt.show()
