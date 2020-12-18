import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

n = 100
m = 300

np.random.seed(1)
A = np.random.rand(m, n)
b = A @ np.ones(n) / 2
c = -np.random.rand(n)
zeros = np.zeros(m)

best_value = 1000
best_t = 1000
objective_values = []
max_violation = []
t_ovs = []
######
x = cp.Variable(n)
objective = cp.Minimize(c @ x)
constraints = [x >= 0, x <= 1, A @ x <= b]
prob = cp.Problem(objective, constraints)
result = prob.solve()
xvalue = x.value
######
for t in np.linspace(0, 1, 101):
    x = np.where(xvalue > t, 1, 0)
    objective_values.append(c.T @ x)
    max_violation.append(max(A @ x - b))
    t_ovs.append(int(t * 100))
    if (np.sum((A @ x - b <= 0)) == m):
        if (best_value > (c @ x)):
            best_value = (c @ x)
            best_t = t
print(best_t)
print(best_value)
print(f'U - L = {best_value} - {result} = {best_value - result}')
plt.plot(t_ovs, objective_values)
plt.plot(t_ovs, max_violation)
plt.legend(['objective values', 'maximum violation'])
plt.xlabel('thresholds * 100')
plt.show()
