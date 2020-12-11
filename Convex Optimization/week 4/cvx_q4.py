import numpy as np
import cvxpy as cp

n=20

np.random.seed(1)

pbar = np.ones((n,1))*0.03 + np.vstack((np.random.rand(n-1,1),0))*0.12

S = np.random.randn(n,n)

S = S.T @ S
S = S/max(abs(np.diag(S)))*.2
S[:,n-1] = np.zeros(n)
S[n-1,:] = np.zeros((n,1)).T
x_unif = np.ones((n,1))/n

reward_min = float(pbar.T @ x_unif)

# ones = np.ones((n,1))

x = cp.Variable(n)
objective = cp.Minimize(cp.quad_form(x, S))
constraints = [x >= 0, cp.sum(x)==1, pbar.T @ x >= reward_min]
prob = cp.Problem(objective, constraints)

result = prob.solve()
print(result)
