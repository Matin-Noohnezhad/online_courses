import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

n = 1

x = cp.Variable(n)
I = np.eye(n)

objective = cp.Minimize(cp.quad_form(x, I)+1)
constraints = [(cp.quad_form(x, I) - 6*x + 8)<=0]  # for question 1
prob = cp.Problem(objective, constraints)

result = prob.solve()
l1 = constraints[0].dual_value[0]
print(result)
print(x.value)
print(constraints[0].dual_value)
################
xi = np.arange(0,5,0.01)
yi = xi**2 + 1
oi = xi**2 - 6*xi + 8
plt.plot(xi, yi)
plt.plot(xi, oi)
plt.show()

legends = []
Lxls = []
for lambd in range(4):
    Lxli = yi + lambd*oi
    plt.plot(xi, Lxli)
    Lxls.append(Lxli)
    legends.append(f'for lambda={lambd}')
plt.legend(legends)
plt.show()
##
plt.show()
