import numpy as np


def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.array x.
    Args:
        x: has to be a numpy.array of dimension m * n.
    Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None
    return np.c_[np.ones(x.shape[0]), x]


if __name__ == "__main__":
    # Example 1:
    x = np.arange(1, 6)
    print(
        f"Test: x=np.arange(1, 6)\nExpected Ouput:\n [[1. 1.]\n [1. 2.]\n [1. 3.]\n [1. 4.]\n [1. 5.]]\nActual Output:\n {add_intercept(x)}\n"
    )

    # Example 2:
    y = np.arange(1, 10).reshape((3, 3))
    print(
        f"Test: y=np.arange(1, 10).reshape((3, 3))\nExpected Ouput:\n [[1. 1. 2. 3.]\n [1. 4. 5. 6.]\n [1. 7. 8. 9.]]\nActual Output:\n {add_intercept(y)}\n"
    )
