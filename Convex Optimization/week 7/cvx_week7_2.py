import numpy as np
from scipy.stats import norm
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib import cm
# import plotly.graph_objects as go

mu1 = 8
mu2 = 20
sigma1 = 6
sigma2 = 17.5
rho = -0.25

# assuming jointly gaussian distribution
mu = mu1 + mu2
sigma = np.sqrt(sigma1 ** 2 + sigma2 ** 2 + 2 * rho * sigma1 * sigma2)
p_loss = norm.cdf(0, mu, sigma)

n = 100
r_min = -30
r_max = 70

# discretize outcomes of R1 and R2
r = np.linspace(r_min, r_max, n)

# marginal distributions
p1 = norm(mu1, sigma1).pdf(r)
p1 = p1/np.sum(p1)
p2 = norm(mu2, sigma2).pdf(r)
p2 = p2/np.sum(p2)

# form mask of region where R1 + R2 <= 0
r1p = r * np.ones((1, n))
r2p = r1p.T
loss_mask = (r1p + r2p <= 0).T

P = cp.Variable((n,n))
objective = cp.Maximize(cp.sum(cp.sum(P[loss_mask])))
constraints = [P >= 0, cp.sum(P,1)==p1, cp.sum(P,0) == p2, (r-mu1).T @ P @ (r-mu2) == rho*sigma1*sigma2]
prob = cp.Problem(objective, constraints)

result = prob.solve()
print(result)

P = P.value
X = np.linspace(r_min,r_max,n)
Y = np.linspace(r_min,r_max,n)
X, Y = np.meshgrid(X, Y)
Z = P

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()
plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.show()