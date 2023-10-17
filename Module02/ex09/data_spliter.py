import numpy as np

def data_spliter(x, y, proportion):
    """
    Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    
    Args:
    x: has to be an numpy.array, a matrix of dimension m * n.
    y: has to be an numpy.array, a vector of dimension m * 1.
    proportion: has to be a float, the proportion of the dataset that will be assigned to the
    training set.

    Return:
    (x_train, x_test, y_train, y_test) as a tuple of numpy.array
    None if x or y is an empty numpy.array.
    None if x and y do not share compatible dimensions.
    None if x, y or proportion is not of expected type.

    Raises:
    This function should not raise any Exception.
    """
    if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(proportion, float)):
        return None

    if x.size == 0 or y.size == 0 or x.shape[0] != y.shape[0]:
        return None

    # Combine x and y
    combined = np.hstack((x, y))

    # Shuffle the combined dataset
    np.random.shuffle(combined)

    # Split the combined dataset back into x and y
    x_shuffled = combined[:, :-1]
    y_shuffled = combined[:, -1].reshape(-1, 1)

    # Calculate the split index
    split_idx = int(x.shape[0] * proportion)

    # Split the shuffled dataset into training and testing sets
    x_train, x_test = x_shuffled[:split_idx], x_shuffled[split_idx:]
    y_train, y_test = y_shuffled[:split_idx], y_shuffled[split_idx:]

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1)) # univariate
    y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
    x_train, x_test, y_train, y_test = data_spliter(x1, y, 0.8)
    print(x_train, x_test, y_train, y_test)
    print()

    x2 = np.array([[1, 42],
                   [300, 10],
                   [59, 1],
                   [300, 59],
                   [10, 42]]) # multivariate
    y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
    x_train, x_test, y_train, y_test = data_spliter(x2, y, 0.5)
    print(x_train, x_test, y_train, y_test)