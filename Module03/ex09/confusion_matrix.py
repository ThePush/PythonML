import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
    y_true = y_true.flatten()
    y_hat = y_hat.flatten()

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_hat)))
    else:
        labels = np.array(labels)

    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    label_index = {label: idx for idx, label in enumerate(labels)}

    for true, hat in zip(y_true, y_hat):
        if true in label_index and hat in label_index:
            matrix[label_index[true], label_index[hat]] += 1

    if df_option:
        return pd.DataFrame(matrix, index=labels, columns=labels)
    else:
        return matrix


if __name__ == "__main__":
    # Examples
    y_hat = np.array(
        ["norminet", "dog", "norminet", "norminet", "dog", "bird"]
    ).reshape((-1, 1))
    y = np.array(["dog", "dog", "norminet", "norminet", "dog", "norminet"]).reshape(
        (-1, 1)
    )

    # Example 1:
    print("Custom implementation:\n", confusion_matrix_(y, y_hat))
    print("Sklearn implementation:\n", confusion_matrix(y, y_hat))

    # Example 2:
    print(
        "Custom implementation:\n",
        confusion_matrix_(y, y_hat, labels=["dog", "norminet"]),
    )
    print(
        "Sklearn implementation:\n",
        confusion_matrix(y, y_hat, labels=["dog", "norminet"]),
    )

    # Example 3:
    print(
        "Custom implementation with DataFrame option:\n",
        confusion_matrix_(y, y_hat, df_option=True),
    )

    # Example 4:
    print(
        "Custom implementation with DataFrame option:\n",
        confusion_matrix_(y, y_hat, labels=["bird", "dog"], df_option=True),
    )