import numpy as np
import math
import sigmoid as sg


def costFunction(theta, X, y):
    # COSTFUNCTION Compute cost and gradient for logistic regression
    #   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    #   parameter for logistic regression and the gradient of the cost
    #   w.r.t. to the parameters.

    # Initialize some useful values
    m = len(y)  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(len(theta))
    # print(theta.shape)
    # print(X.shape)
    # print(y.shape)
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    #
    # _sum = 0
    # for i in range(m):
    #     _sum = _sum + ((-y[i]) * (math.log(sg.sigmoid(np.dot(X[i], np.reshape(theta, (len(theta), 1)))))) - (1 - y[i]) * (
    #         math.log(1 - sg.sigmoid(np.dot(X[i], np.reshape(theta, (len(theta), 1)))))))
    #
    # J = _sum / m
    thetaX = np.dot(X, np.reshape(theta, (len(theta), 1)))
    hthetaX = sg.sigmoid(thetaX)
    _sum = np.dot(np.transpose(y), np.log(hthetaX)) + np.dot(1 - np.transpose(y), np.log(1 - hthetaX))
    J = -1 * _sum / m
    # ...
    # l = len(theta)
    # for j in range(l):
    #     _sum = 0
    #     for i in range(m):
    #         _sum = _sum + ((sg.sigmoid(np.dot(X[i], np.reshape(theta, (len(theta), 1))))) - y[i]) * X[i, j]
    #     grad[j] = _sum / m
    grad = np.dot(np.transpose(X) , (hthetaX - y))/m
    # ...
    # =============================================================

    return [J, grad]
