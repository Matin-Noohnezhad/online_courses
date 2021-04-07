from computeCostMulti import computeCostMulti
import numpy as np


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    # GRADIENTDESCENTMULTI Performs gradient descent to learn theta
    #   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
    #   taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    m = len(y);  # number of training examples
    J_history = {}

    for iter in range(1, num_iters + 1):

        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta. 
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCostMulti) and gradient here.
        #
        temp_theta = np.zeros((len(theta), 1))
        for j in range(len(theta)):
            sigma = 0
            for i in range(m):
                sigma = sigma + np.subtract(np.dot(X[i], theta), y[i]) * X[i, j]
            temp_theta[j] = theta[j] - alpha * (1 / m) * sigma
        for j in range(len(theta)):
            theta[j] = temp_theta[j]

        # Save the cost J in every iteration
        J_history[iter] = computeCostMulti(X, y, theta)
    return [theta, J_history]
