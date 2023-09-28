import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression

# Load the dataset
data = pd.read_csv("are_blue_pills_magics.csv")
X = data["Micrograms"].values.reshape(-1, 1)
y = data["Score"].values.reshape(-1, 1)

# Initialize variables
degrees = list(range(1, 7))
mse_scores = []
models = []

# Train and evaluate models
for degree in degrees:
    X_poly = add_polynomial_features(X, degree)

    # Initialize thetas
    if degree == 1:
        thetas = np.array([[86], [-8]]).reshape(-1, 1)
    elif degree == 2:
        thetas = np.array([[54], [10], [-2]]).reshape(-1, 1)
    elif degree == 3:
        thetas = np.array([[50], [35], [-13], [1]]).reshape(-1, 1)
    elif degree == 4:
        thetas = np.array([[-20], [160], [-80], [10], [-1]]).reshape(-1, 1)
    elif degree == 5:
        thetas = np.array([[1140], [-1850], [1110], [-305], [40], [-2]]).reshape(-1, 1)
    elif degree == 6:
        thetas = np.array(
            [[9110], [-18015], [13400], [-4935], [966], [-96.4], [3.86]]
        ).reshape(-1, 1)
    else:
        thetas = np.zeros((degree + 1, 1))

    model = MyLinearRegression(
        thetas=thetas, alpha=1 / (10000 ** (degree + 1)), max_iter=10000
    )
    model.fit_(X_poly, y)
    y_pred = model.predict_(X_poly)

    mse = model.mse_(y, y_pred)
    mse_scores.append(mse)
    models.append(model)

    print(f"Degree: {degree}, MSE: {mse}")
    print(f"Thetas: {model.thetas.flatten()}\n")

# Plot MSE scores
plt.figure(figsize=(10, 5))
plt.bar(degrees, mse_scores)
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("MSE vs Polynomial Degree")
plt.show()

# Plot models, data points, and predicted points
plt.figure(figsize=(10, 5))
plt.scatter(X, y, label="Dataset Points", color="red", alpha=0.5)

X_range = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
for degree, model in zip(degrees, models):
    X_poly_range = add_polynomial_features(X_range, degree)
    y_pred_range = model.predict_(X_poly_range)
    plt.plot(X_range, y_pred_range, label=f"Degree {degree}")

    # Plot predicted points
    X_poly = add_polynomial_features(X, degree)
    y_pred = model.predict_(X_poly)
    plt.scatter(
        X, y_pred, marker="x", s=20, label=f"Predicted Points (Degree {degree})"
    )
    #if degree == 3:
    #    break

plt.xlabel("Micrograms")
plt.ylabel("Score")
plt.title("Polynomial Models, Data Points, and Predicted Points")
plt.legend()
plt.show()
