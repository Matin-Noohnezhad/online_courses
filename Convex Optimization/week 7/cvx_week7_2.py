import numpy as np
import cvxpy as cp


mu = cp.Variable(2)
sigma = cp.Variable(2) #standard deviation
ro = cp.Variable(1) #correlation

r = np.arange(-30, 71)
