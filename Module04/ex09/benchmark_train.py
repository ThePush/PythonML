import pickle
import numpy as np
import pandas as pd
from my_logistic_regression import MyLogisticRegression
from data_spliter import data_spliter
from polynomial_model import add_polynomial_features
from z_score import zscore
from other_metrics import f1_score_


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

    num_classes = len(np.unique(y_train))
    lambdas = np.linspace(0, 1, 11)
    models = []
    for lambda_ in lambdas:
        classifiers = []
        for c in range(num_classes):
            theta = np.zeros((X_train.shape[1] + 1, 1))
            model = MyLogisticRegression(
                theta, alpha=0.001, max_iter=1000, penalty="l2", lambda_=lambda_
            )
            y_binary = (y_train == c).astype(int)
            model.fit_(X_train, y_binary)
            classifiers.append(model)
        models.append(classifiers)

    f1_scores = []
    for classifiers in models:
        y_hat = []
        for model in classifiers:
            y_hat.append(model.predict_(x_cross_test))
        y_hat = np.argmax(np.hstack(y_hat), axis=1).reshape(-1, 1)
        f1_scores.append(f1_score_(y_cross_test, y_hat))

    best_model_idx = np.argmax(f1_scores)
    best_classifiers = models[best_model_idx]

    with open("models.pkl", "wb") as f:
        pickle.dump(models, f)

    for i, f1_score in enumerate(f1_scores):
        print(f"lambda = {lambdas[i]:.1f}: f1 score = {f1_score}")
