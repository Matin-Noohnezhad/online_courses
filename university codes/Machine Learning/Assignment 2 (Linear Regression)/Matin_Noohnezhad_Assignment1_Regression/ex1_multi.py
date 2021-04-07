import numpy as np

from plotData import plotData
from gradientDescent import gradientDescent
from computeCost import computeCost
from gradientDescentMulti import gradientDescentMulti
from computeCostMulti import computeCostMulti
from featureNormalize import featureNormalize
from normalEqn import normalEqn
## Machine Learning Online Class
#  Exercise 1: Linear regression with multiple variables
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear regression exercise. 
#
#  You will need to complete the following functions in this 
#  exericse:
#
#     plotData.m
#     gradientDescent.m
#     computeCost.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
#
#  For this part of the exercise, you will need to change some
#  parts of the code below for various experiments (e.g., changing
#  learning rates).
#

## Initialization

## ================ Part 1: Feature Normalization ================



print ('Loading data ...')

## Load Data
lines = open('ex1data2.txt', 'rb').readlines()
data = np.array([[float(f) for f in l.decode('UTF-8').split(',')] for l in lines])
X = np.array(data[:, [0, 1]])
y = np.array(data[:, 2])
m = len(y)

# Print out some data points
# print ('First 10 examples from the dataset:')
# print (X[range(0, 10),:], y[range(0, 10)])
print ('Program paused. Press enter to continue.')

# Scale features and set them to zero mean
print ('Normalizing Features ...')

[X, mu, sigma] = featureNormalize(X)

# Add intercept term to X
X = np.hstack((np.ones((m, 1)), X))


## ================ Part 2: Gradient Descent ================

# ====================== YOUR CODE HERE ======================
# Instructions: We have provided you with the following starter
#               code that runs gradient descent with a particular
#               learning rate (alpha). 
#
#               Your task is to first make sure that your functions - 
#               computeCost and gradientDescent already work with 
#               this starter code and support multiple variables.
#
#               After that, try running gradient descent with 
#               different values of alpha and see which one gives
#               you the best result.
#
#               Finally, you should complete the code at the end
#               to predict the price of a 1650 sq-ft, 3 br house.
#
# Hint: By using the 'hold on' command, you can plot multiple
#       graphs on the same figure.
#
# Hint: At prediction, make sure you do the same feature normalization.
#

print ('Running gradient descent ...')

# Choose some alpha value
alpha = 0.15
num_iters = 100

# Init Theta and Run Gradient Descent 
theta = np.zeros((3, 1))
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)


# Plot the convergence graph
iter_nums = J_history.keys()
costs = [J_history[num] for num in iter_nums]

plotData(np.array(range(1,num_iters+1)), costs, '*', xl = 'Number of iterations', yl = 'Cost J')#, '-b', 'LineWidth', 2)

# Display gradient descent's result
print ('Theta computed from gradient descent: ')
print (' #f ', theta)
print ('')

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.
price = 0 # You should change this

area = ((1650 - mu[0,0]) / sigma[0,0]) 
num = ((3 - mu[0,1]) / sigma[0,1]) 
price = np.dot([1, area, num] , theta) 

# ============================================================

print ('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ', price)

print ('Program paused. Press enter to continue.')


## ================ Part 3: Normal Equations ================

print ('Solving with normal equations...')

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form 
#               solution for linear regression using the normal
#               equations. You should complete the code in 
#               normalEqn.m
#
#               After doing so, you should complete this code 
#               to predict the price of a 1650 sq-ft, 3 br house.
#

## Load Data
lines = open('ex1data2.txt', 'rb').readlines()
data = np.array([[float(f) for f in l.decode('UTF-8').split(',')] for l in lines])

X = data[:, 0:2]
y = data[:, 2]
m = len(y)
# Add intercept term to X
X = np.hstack((np.ones((m, 1)), X))

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print ('Theta computed from the normal equations: ')
print (' #f ', theta)
print ('')


# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
price = 0 # You should change this

x = np.array([1, 1650, 3])
print (theta)
# print('matin' , np.shape(x))
# print('matin' , np.shape(theta))
price = np.dot(x, theta)

# ============================================================

print ('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):', price)

