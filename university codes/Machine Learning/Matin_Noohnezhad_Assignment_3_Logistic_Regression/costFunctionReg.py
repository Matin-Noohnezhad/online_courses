import numpy as np
from costFunction import *
import math


def costFunctionReg(theta, X, y, lambda_):
    # COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    #   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters.

    # Initialize some useful values
    m = len(y)  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(len(theta))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    cf = costFunction(theta, X, y)
    J = cf[0]
    sigma = np.sum(math.pow(t, 2) for t in theta)
    J = J + ((lambda_ * sigma) / (2 * m))
    # l = len(theta)
    # for i in range(1, l):
    grad = cf[1] + (lambda_ * theta / m)
    grad[0] = (cf[1])[0]
    # ...
    # ...
    # =============================================================
    return [J, grad]
