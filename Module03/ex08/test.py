import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from other_metrics import accuracy_score_, precision_score_, recall_score_, f1_score_

# Example 1:
y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))

# Accuracy
print("Accuracy (custom implementation):", accuracy_score_(y, y_hat))
print("Accuracy (sklearn implementation):", accuracy_score(y, y_hat))

# Precision
print("Precision (custom implementation):", precision_score_(y, y_hat))
print("Precision (sklearn implementation):", precision_score(y, y_hat))

# Recall
print("Recall (custom implementation):", recall_score_(y, y_hat))
print("Recall (sklearn implementation):", recall_score(y, y_hat))

# F1-score
print("F1-score (custom implementation):", f1_score_(y, y_hat))
print("F1-score (sklearn implementation):", f1_score(y, y_hat))

# Example 2:
y_hat = np.array(
    ["norminet", "dog", "norminet", "norminet", "dog", "dog", "dog", "dog"]
)
y = np.array(
    ["dog", "dog", "norminet", "norminet", "dog", "norminet", "dog", "norminet"]
)

# Accuracy
print("Accuracy (custom implementation):", accuracy_score_(y, y_hat))
print("Accuracy (sklearn implementation):", accuracy_score(y, y_hat))

# Precision
print("Precision (custom implementation):", precision_score_(y, y_hat, pos_label="dog"))
print("Precision (sklearn implementation):", precision_score(y, y_hat, pos_label="dog"))

# Recall
print("Recall (custom implementation):", recall_score_(y, y_hat, pos_label="dog"))
print("Recall (sklearn implementation):", recall_score(y, y_hat, pos_label="dog"))

# F1-score
print("F1-score (custom implementation):", f1_score_(y, y_hat, pos_label="dog"))
print("F1-score (sklearn implementation):", f1_score(y, y_hat, pos_label="dog"))

# Example 3:
y_hat = np.array(
    ["norminet", "dog", "norminet", "norminet", "dog", "dog", "dog", "dog"]
)
y = np.array(
    ["dog", "dog", "norminet", "norminet", "dog", "norminet", "dog", "norminet"]
)

# Precision
print(
    "Precision (custom implementation):",
    precision_score_(y, y_hat, pos_label="norminet"),
)
print(
    "Precision (sklearn implementation):",
    precision_score(y, y_hat, pos_label="norminet"),
)

# Recall
print("Recall (custom implementation):", recall_score_(y, y_hat, pos_label="norminet"))
print("Recall (sklearn implementation):", recall_score(y, y_hat, pos_label="norminet"))

# F1-score
print("F1-score (custom implementation):", f1_score_(y, y_hat, pos_label="norminet"))
print("F1-score (sklearn implementation):", f1_score(y, y_hat, pos_label="norminet"))
