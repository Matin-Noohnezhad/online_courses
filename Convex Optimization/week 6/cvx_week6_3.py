import cvxpy as cp
import cvxopt
import numpy as np
import math

k = 201
t = [(-3 + 6 * (i - 1) / (k - 1)) for i in range(1, k + 1)]
t = np.array(t)
y = np.exp(t)
t2 = t ** 2
t_powers = np.vstack((np.ones(t.shape), t, t2))

n = 3
# objective = cp.Minimize(cp.norm(numerator / denominator - y , 'inf'))
u = math.exp(3)
l = 0
bisection_tolerance = 1e-3
a_opt = None
b_opt = None
obj_val_opt = None
while (u - l >= bisection_tolerance):
    gamma = (u + l) / 2
    #
    a = cp.Variable(n)
    b = cp.Variable(n - 1)
    bb = cp.hstack((1, b))
    a_t_powers = a @ t_powers
    b_t_powers = bb @ t_powers
    objective = cp.Minimize(0)
    constraints = [(a_t_powers - y @ b_t_powers) >= (gamma * b_t_powers)]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    if prob.status == cp.OPTIMAL:
        a_opt = a
        b_opt = b
        u = gamma
        obj_val_opt = gamma
        print('u is gamma')
    else:
        l = gamma
        print('l is gamma')


print(a_opt.value)
print(b_opt.value)
print(obj_val_opt)
