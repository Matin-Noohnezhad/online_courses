from plotData import plotData
from gradientDescent import gradientDescent
from computeCost import computeCost
from gradientDescentMulti import gradientDescentMulti
from computeCostMulti import computeCostMulti
from featureNormalize import featureNormalize
from normalEqn import normalEqn
import numpy as np
from pylab import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

## Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     plotData.py
#     gradientDescent.py
#     computeCost.py
#     gradientDescentMulti.py
#     computeCostMulti.py
#     featureNormalize.py
#     normalEqn.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#

## Initialization

## ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.m 
print ('Running warmUpExercise ... \n')
print ('5x5 Identity Matrix: \n')
#warmUpExercise()

print ('Program paused. Press enter to continue.\n')



## ======================= Part 2: Plotting =======================
print ('Plotting Data ...\n')
data = open('ex1data1.txt', 'rb')
lines = data.readlines()
lines = [l.decode('UTF-8').split(',') for l in lines]

X = [float(l[0]) for l in lines]
y = np.array([float(l[1]) for l in lines])
m = len(y) # number of training examples

# Plot Data
# Note: You have to complete the code in plotData.m
plotData(X, y, '*')
print('-------------------------------')
print(np.shape(X))
print(np.shape(y))

print ('Program paused. Press enter to continue.\n')


## =================== Part 3: Gradient descent ===================
print ('Running Gradient Descent ...\n')

X = np.array([[1.0, x] for x in X]) # Add a column of ones to x
theta = np.zeros((2, 1)) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# compute and display initial cost
j = computeCost(X, y, theta)

print('cost is: ', j )

# run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations)[0]

# print theta to screen
print ('Theta found by gradient descent: ')
print (theta[0], theta[1])

# Plot the linear fit
# keep previous plot visible
plotData(X[: , 1], np.dot(X, theta), '-')
print('-------------------------------')
print(np.shape(X[: , 1]))
print(np.shape(np.dot(X, theta)))
#legend('Training data', 'Linear regression')
#hold off # don't overlay any more plots on this figure

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(np.array([1, 3.5]), theta)
print ('For population = 35,000, we predict a profit of #f\n', predict1*10000)
predict2 = np.dot(np.array([1, 7]), theta)
print ('For population = 70,000, we predict a profit of #f\n', predict2*10000)
print ('Program paused. Press enter to continue.\n')

## ============= Part 4: Visualizing J(theta_0, theta_1) =============
#print 'Visualizing J(theta_0, theta_1) ...\n'

# Grid over which we will calculate J
#theta0_vals = np.linspace(-10, 10, num=100)
#theta1_vals = np.linspace(-1, 4, num=100)

# initialize J_vals to a matrix of 0's
#J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals

#J_vals = np.array([computeCost(X, y, np.array([t0, t1])) for t0,t1 in zip(np.ravel(theta0_vals), np.ravel(theta1_vals))])

# Because of the way meshgrids work in the surf command, we need to 
# transpose J_vals before calling surf, or else the axes will be flipped
#J_vals = J_vals.transpose()
# Surface plot
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(theta0_vals, theta1_vals, J_vals)
#plt.show()

#xlabel('\theta_0') ylabel('\theta_1')

# Contour plot

# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
#ax.contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
# xlabel('\theta_0') ylabel('\theta_1')

# plotData(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2)



