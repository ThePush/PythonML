import numpy as np

def minmax(x):
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None

    min_x = np.min(x)
    max_x = np.max(x)
    normalized_x = (x - min_x) / (max_x - min_x)

    return normalized_x


if __name__ == "__main__":
    # Example 1:
    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
    print(minmax(X))
    # Output:
    # array([[0.58333333],
    #        [1.        ],
    #        [0.33333333],
    #        [0.77777778],
    #        [0.91666667],
    #        [0.66666667],
    #        [0.        ]])

    # Example 2:
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    print(minmax(Y))
    # Output:
    # array([[0.63636364],
    #        [1.        ],
    #        [0.18181818],
    #        [0.72727273],
    #        [0.93939394],
    #        [0.6969697 ],
    #        [0.        ]])