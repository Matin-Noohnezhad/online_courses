from sigmoid import sigmoid
from costFunction import costFunction
from predict import predict
from costFunctionReg import costFunctionReg
from plotData import plotData
from mapFeature import mapFeature
import numpy as np
from plotDecisionBoundary import plotDecisionBoundary

## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization

## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

lines = open('ex2data2.txt', 'rb').readlines()
data = np.array([[float(f) for f in l.decode('UTF-8').split(',')] for l in lines])
print(data.shape)
X = data[:, [0, 1]] # m * 2
y = data[:, [2]]; # m * 1

plotData(X, y, xl='Microchip Test 1', yl='Microchip Test 2')

# Put some labels 
# Labels and Legend

# Specified in plot order
#legend('y = 1', 'y = 0')



## =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic 
#  regression to classify the data points. 
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled

X = mapFeature(X[:, 0], X[:, 1])

# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1], 1))

# Set regularization parameter lambda to 1
lambda_ = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda_)

print ('Cost at initial theta (zeros): #f', cost)

print ('Program paused. Press enter to continue.')


## ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and 
#  see how regularization affects the decision coundart
#
#  Try the following values of lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?
#

# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1], 1))

# Set regularization parameter lambda to 1 (you should vary this)
lambda_ = 1

# Set Options
#options = optimset('GradObj', 'on', 'MaxIter', 400)

# Optimize
#[theta, J, exit_flag] = ...
#fminunc(@(t)(costFunctionReg(t, X, y, lambda_)), initial_theta, options)
from scipy.optimize import fmin

theta = fmin(lambda t: (costFunctionReg(t, X, y,lambda_)[0]), initial_theta, xtol=1e-8)

# Plot Boundary
#plotDecisionBoundary(theta, X, y)

#title(sprintf('lambda = #g', lambda_))

# Labels and Legend
#xlabel('Microchip Test 1')
#ylabel('Microchip Test 2')

#legend('y = 1', 'y = 0', 'Decision boundary')


# Compute accuracy on our training set
p = predict(theta, X)
print ('Train Accuracy: ', np.mean([1.0 if i==j else 0.0 for (i, j) in zip(p, y)]) * 100)



