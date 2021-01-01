import numpy as np
import cvxpy as cp

n = 2
ds = [[0,0],[0,-0.1],[0,0.1],[-0.1,0],[-0.1,-0.1],[-0.1,0.1],[0.1,0],[0.1,-0.1],[0.1,0.1]]

A = np.array([[1, 2], [1, -4], [-1, -1]])
b = np.array([-2, -3, 5])
Q = np.array([[1, -1 / 2], [-1 / 2, 2]])
f = np.array([[-1, 0]])

x = cp.Variable(n)

objective = cp.Minimize(cp.quad_form(x, Q) - f @ x)
constraints = [A @ x <= b]  # for question 1
prob = cp.Problem(objective, constraints)

result1 = prob.solve()
l1 = constraints[0].dual_value[0]
l2 = constraints[0].dual_value[1]
################
for d1,d2 in ds:
    A = np.array([[1, 2], [1, -4], [-1, -1]])
    b = np.array([-2 + d1, -3 + d2, 5])
    Q = np.array([[1, -1 / 2], [-1 / 2, 2]])
    f = np.array([[-1, 0]])

    x = cp.Variable(n)

    objective = cp.Minimize(cp.quad_form(x, Q) - f @ x)
    constraints = [A @ x <= b]  # for question 1
    prob = cp.Problem(objective, constraints)

    result = prob.solve()
    pstar_pred = result1 - l1 * d1 - l2 * d2
    print(result)
    print(f'for d1={d1} , d2={d2} --> p* - p*pred is {result - pstar_pred}')


