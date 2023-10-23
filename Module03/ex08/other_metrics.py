import numpy as np


def accuracy_score_(y, y_hat):
    if y.size == 0 or y_hat.size == 0:
        return None

    return np.mean(y == y_hat)


def precision_score_(y, y_hat, pos_label=1):
    if y.size == 0 or y_hat.size == 0:
        return None

    tp = np.sum((y == pos_label) & (y_hat == pos_label))
    fp = np.sum((y != pos_label) & (y_hat == pos_label))

    return tp / (tp + fp)


def recall_score_(y, y_hat, pos_label=1):
    if y.size == 0 or y_hat.size == 0:
        return None

    tp = np.sum((y == pos_label) & (y_hat == pos_label))
    fn = np.sum((y == pos_label) & (y_hat != pos_label))

    return tp / (tp + fn)


def f1_score_(y, y_hat, pos_label=1):
    if y.size == 0 or y_hat.size == 0:
        return None

    precision = precision_score_(y, y_hat, pos_label)
    recall = recall_score_(y, y_hat, pos_label)

    return 2 * precision * recall / (precision + recall)
