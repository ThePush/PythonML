import numpy as np
import pandas as pd
import argparse
import sys
from my_logistic_regression import MyLogisticRegression as MyLR
import matplotlib.pyplot as plt
from data_spliter import data_spliter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zipcode of your favourite planet.")
    parser.add_argument("-zipcode", type=int)

    args = parser.parse_args()
    if args.zipcode is None or not 0 <= args.zipcode <= 3:
        parser.print_usage()
        exit(1)

    planet_zipcode = args.zipcode

    # Load data
    data_x = pd.read_csv("solar_system_census.csv")
    data_y = pd.read_csv("solar_system_census_planets.csv")
    features = ["Weight", "Height", "Bone Density"]
    planets = ["Venus", "Earth", "Mars", "Asteroid Belt"]
    data_x.drop("Unnamed: 0", axis=1, inplace=True)
    data_y.drop("Unnamed: 0", axis=1, inplace=True)

    # Preprocess data
    X = data_x.to_numpy()
    y = (data_y["Origin"] == planet_zipcode).astype(int).to_numpy().reshape(-1, 1)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = data_spliter(X, y, 0.8)

    # Train logistic regression model
    theta = np.random.randn(X.shape[1] + 1, 1)
    mylr = MyLR(theta, alpha=0.0001, max_iter=500_000)
    mylr.fit_(X_train, y_train)

    # Make predictions
    y_pred = (mylr.predict_(X_test) >= 0.5).astype(int)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print("Correct predictions:", np.sum(y_pred == y_test), "/", len(y_pred))
    print("Accuracy:", accuracy)

    # Plot scatter plots
    for idx, feature in enumerate(features):
        plt.xlabel(feature)
        plt.ylabel("Zipcode")
        plt.yticks(
            range(0, 2),
            ["Not " + planets[planet_zipcode], planets[planet_zipcode]],
        )
        column = X_test[:, idx]
        plt.scatter(column, y_test, label="Real values", s=200)
        plt.scatter(column, y_pred, label="Predictions", s=50)
        plt.legend(loc="best")
        plt.show()
