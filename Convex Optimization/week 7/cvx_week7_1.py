# clear all; close all;
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# create problem data
N = 100
# create an increasing input signal
ytrue = np.zeros((N, 1))
ytrue[:40] = 0.1
ytrue[49] = 2
ytrue[69:80] = 0.15
ytrue[79] = 1
ytrue = np.cumsum(ytrue)

# pass the increasing input through a moving-average filter
# and add Gaussian noise
h = np.array([1, -0.85, 0.7, -0.3])
k = len(h)
# print(h.shape)
# print(xtrue.shape)
yhat = np.convolve(h, ytrue)
print(yhat.shape)
y = yhat[:-3] + np.array(
    [-0.43, -1.7, 0.13, 0.29, -1.1, 1.2, 1.2, -0.038, 0.33, 0.17, -0.19, 0.73, -0.59, 2.2, -0.14, 0.11, 1.1, 0.059,
     -0.096, -0.83, 0.29, -1.3, 0.71, 1.6, -0.69, 0.86, 1.3, -1.6, -1.4, 0.57, -0.4, 0.69, 0.82, 0.71, 1.3, 0.67, 1.2,
     -1.2, -0.02, -0.16, -1.6, 0.26, -1.1, 1.4, -0.81, 0.53, 0.22, -0.92, -2.2, -0.059, -1, 0.61, 0.51, 1.7, 0.59,
     -0.64, 0.38, -1, -0.02, -0.048, 4.3e-05, -0.32, 1.1, -1.9, 0.43, 0.9, 0.73, 0.58, 0.04, 0.68, 0.57, -0.26, -0.38,
     -0.3, -1.5, -0.23, 0.12, 0.31, 1.4, -0.35, 0.62, 0.8, 0.94, -0.99, 0.21, 0.24, -1, -0.74, 1.1, -0.13, 0.39, 0.088,
     -0.64, -0.56, 0.44, -0.95, 0.78, 0.57, -0.82, -0.27])
print(y.shape)

w = cp.Variable(2)
t = np.arange(N)
t = np.vstack((np.ones(N), t))
t = t.reshape(2, 100)
objective = cp.Minimize(cp.sum_squares(w @ t - y))
constraints = [w >= 0, w @ t >= 0]
prob = cp.Problem(objective, constraints)
# prob = cp.Problem(objective)
error = prob.solve()
print(w.value)
print('error is ', error)

y_approx = w.value @ t
plt.plot(y_approx)
plt.plot(ytrue)
plt.plot(y)
plt.show()
