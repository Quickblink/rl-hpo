from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF #DotProduct, WhiteKernel

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal


x = np.linspace(0, 1, 100)
x1 = np.linspace(-5, 0, 100)
x2 = np.linspace(0, 5, 100)

K = norm.pdf(10*np.abs(np.subtract(*np.meshgrid(x, x))))
plt.figure()
for _ in range(10):
        plt.plot(x, multivariate_normal.rvs(np.zeros(100, dtype=np.float), K))

#X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
kernel = RBF(0.1)#DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel,
        random_state=0, optimizer=None)#.fit(X, y)

plt.figure()
for i in range(10):
        y1 = gpr.sample_y(x[:, None], random_state=i).squeeze()
        #gpr.fit(x1[:, None], y1)
        plt.plot(x, y1)
        #plt.plot(x2, gpr.sample_y(x2[:, None], random_state=i).squeeze())


plt.show()





#print(gpr.score(X, y))

#print(gpr.predict(X[:2,:], return_std=True))

print(gpr.get_params())