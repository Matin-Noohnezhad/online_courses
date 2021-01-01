import cvxpy as cp
import numpy as np


A = np.array([2, 1])
B = np.array([1, 3])
C = np.array([1, 9])


# Construct the problem.
x = cp.Variable(2)
objective = cp.Minimize(C@cp.power(x, 2))
constraints = [0 <= x, 1 <= A@x, 1 <= B@x]
prob = cp.Problem(objective, constraints)

# The optimal objective is returned by prob.solve().
result = prob.solve()
# The optimal value for x is stored in x.value.
print(x.value)
# The optimal Lagrange multiplier for a constraint
# is stored in constraint.dual_value.
# print(constraints[0].dual_value)
print(result)


