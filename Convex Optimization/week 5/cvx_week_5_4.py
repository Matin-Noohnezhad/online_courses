import numpy as np
import cvxpy as cp

r = 1.05  # risk-free rate
m = 200  # scenarios
n = 7  # assets
V = np.zeros((m, n))  # value/payoff matrix
V[:, 0] = r  # risk-free asset
V[:, 1] = np.linspace(0.5, 2, m)
# underlying% the four exchange traded options:
V[:, 2] = np.maximum(V[:, 1] - 1.1, 0)
V[:, 3] = np.maximum(V[:, 1] - 1.2, 0)
V[:, 4] = np.maximum(0.8 - V[:, 1], 0)
V[:, 5] = np.maximum(0.7 - V[:, 1], 0)
# collar option:
F = 0.9
C = 1.15
V[:, 6] = np.minimum(np.maximum(V[:, 1], F), C) - 1
p = np.array([1, 1, 0.06, 0.03, 0.02, 0.01])  # asset prices (from exchange)

y = cp.Variable(m)
p_collar = cp.Variable()
P = cp.hstack((p, p_collar))

objective = cp.Minimize(p_collar)
constraints = [y >= 0, V.T @ y == P]
prob = cp.Problem(objective, constraints)
lb = prob.solve()

objective = cp.Maximize(p_collar)
constraints = [y >= 0, V.T @ y == P]
prob = cp.Problem(objective, constraints)
ub = prob.solve()

print('lb=', lb, '\nub=', ub)
