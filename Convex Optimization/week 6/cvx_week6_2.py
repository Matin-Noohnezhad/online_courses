# data for censored fitting problem.
import math
import numpy as np
import cvxpy as cp

n = 20  # dimension of x's
M = 25  # number of non-censored data points
K = 100  # total number of points

c_true = np.loadtxt("c_true.csv", delimiter=",")
x = np.loadtxt("data.csv", delimiter=",")
noise = np.loadtxt("noise.csv", delimiter=",")
y = x @ c_true + 0.1 * (math.sqrt(n)) * noise

# # Reorder measurements, then censor
sort_ind = np.argsort(y)
y = y[sort_ind]
x = x[sort_ind]
D = (y[M] + y[M + 1]) / 2
y = y[:M]
print(y.shape)
print(x.shape)

c_hat = cp.Variable(n)
y_censored = cp.Variable(K - M)
Y = cp.hstack((y, y_censored))
objective = cp.Minimize(cp.sum_squares(x @ c_hat - Y))
constraints = [y_censored >= D]
prob = cp.Problem(objective, constraints)
prob.solve()

print(c_hat.value)
# print(y_censored.value)

residual = np.linalg.norm(c_true - c_hat.value) / np.linalg.norm(c_true)
print('residual for c_hat', residual)

print('#########################################################################')

c_hat = cp.Variable(n)
x = x[:M]
objective = cp.Minimize(cp.sum_squares(x @ c_hat - y))
prob = cp.Problem(objective)
prob.solve()

print(c_hat.value)
# print(y_censored.value)

residual = np.linalg.norm(c_true - c_hat.value) / np.linalg.norm(c_true)
print('residual for c_ls', residual)

