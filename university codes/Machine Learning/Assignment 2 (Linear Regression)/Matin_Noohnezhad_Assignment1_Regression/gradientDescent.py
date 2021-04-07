import numpy as np
from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    # GRADIENTDESCENT Performs gradient descent to learn theta
    #   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
    #   taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values

    m = len(y)  # number of training examples
    J_history = {}

    m = len(y)
    for iter in range(1, num_iters + 1):

        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
        temp_theta = np.zeros((len(theta) , 1))
        for j in range(len(theta)):
            sigma = 0
            for i in range(m):
                sigma =sigma + np.subtract(np.dot(X[i],theta) , y[i])*X[i,j]
            temp_theta[j] = theta[j] - alpha*(1/m)*sigma
        if(iter<200):
            print(temp_theta , iter)
        for j in range(len(theta)):
            theta[j] = temp_theta[j]
        # ============================================================

        # Save the cost J in every iteration
        J_history[iter] = computeCost(X, y, theta);
    # print(J_history)
    return [theta, J_history]
