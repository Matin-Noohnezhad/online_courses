import numpy as np


def mapFeature(X1, X2):
    # MAPFEATURE Feature mapping function to polynomial features
    #
    #   MAPFEATURE(X1, X2) maps the two input features
    #   to quadratic features used in the regularization exercise.
    #
    #   Returns a new feature array with more features, comprising of
    #   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    #
    #   Inputs X1, X2 must be the same size
    #
    X1 = X1.reshape(len(X1), 1)
    X2 = X2.reshape(len(X2), 1)
    degree = 6
    out = np.ones((X1.shape[0], 1))
    out = np.hstack((out, X1))
    out = np.hstack((out, X2))

    # ...
    for i in range(2, degree + 1):
        for j in range(i + 1):
            out = np.hstack((out, np.multiply(np.power(X1, j), np.power(X2, (i - j)))))
    # ...
    return out
