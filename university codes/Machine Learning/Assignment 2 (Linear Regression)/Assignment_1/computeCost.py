import numpy as np


def computeCost(X, y, theta):
    # COMPUTECOST Compute cost for linear regression
    #   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    #   parameter for linear regression to fit the data points in X and y

    # Initialize some useful values
    # print(X)
    m = len(y)  # number of training examples

    # You need to return the following variables correctly
    J = 0
    # print(np.shape(y))
    # print(np.shape(np.dot(X , theta)))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.
    diff = np.subtract(np.reshape(y, (m, 1)), (np.dot(X, theta)))
    sigma = np.sum(np.power(diff, 2))
    J = 1 / (2 * m) * sigma

    return J

# =========================================================================
