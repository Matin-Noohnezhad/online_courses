from sigmoid import sigmoid
from costFunction import costFunction
from predict import predict
from costFunctionReg import costFunctionReg
from plotData import plotData
from plotDecisionBoundary import plotDecisionBoundary
from pylab import *
import numpy as np
from scipy.optimize import fmin

## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions 
#  in this exericse:
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

lines = open('ex2data1.txt', 'rb').readlines()
data = np.array([[float(f) for f in l.decode('UTF-8').split(',')] for l in lines])
print(shape(data))
X = data[:, [0, 1]] # m * 2
y = data[:, [2]]; # m * 1

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

print ('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

plotData(X, y, xl = 'Exam 1 score', yl='Exam 2 score')


print ('Program paused. Press enter to continue.')



## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = X.shape;

# Add intercept term to x and X_test
X = np.hstack((np.ones((m, 1)), X))

# Initialize fitting parameters
initial_theta = np.zeros((n + 1, 1))
[cost, grad] = costFunction(initial_theta, X, y)
print ('Cost at initial theta (zeros):', cost)
print ('Gradient at initial theta (zeros):')
print (' #f ', grad)

print ('Program paused. Press enter to continue.')


## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.

#  Set options for fminunc

#options = optimset('GradObj', 'on', 'MaxIter', 400);

#  Run fminunc to obtain the optimal theta
#  This function will return theta and the cost 
print (initial_theta)
theta = fmin(lambda t: (costFunction(t, X, y)[0]), initial_theta, xtol=1e-8);

# Print theta to screen
#print 'Cost at theta found by fminunc: #f', cost
print ('theta: ')
print (' #f ', theta)

# # Plot Boundary
plotDecisionBoundary(theta, X, y);

# # Put some labels 
# # Labels and Legend
# xlabel('Exam 1 score')
# ylabel('Exam 2 score')

# # Specified in plot order
# legend('Admitted', 'Not admitted')

# print 'Program paused. Press enter to continue.'


## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 20 on exam 1 and 
#  score 80 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of 
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 

x = np.array([1, 45, 85])
prob = sigmoid(np.dot(x, theta));
print ('For a student with scores 45 and 85, we predict an admission probability of ', prob)

predict(theta, x)

# prob = sigmoid([1 80 90] * theta); 
# fprint(['For a student with scores 80 and 90, we predict

# Compute accuracy on our training set
p = predict(theta, X)

print ('Train Accuracy: #f', np.mean([1.0 if i==j else 0.0 for (i,j) in zip(p,y)]) * 100)

print ('Program paused. Press enter to continue.')


