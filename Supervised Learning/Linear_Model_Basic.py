# %% Import Packages
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
import pandas as pd

# %% normal linear least square
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
print(reg.coef_, reg.intercept_)

# %% Ridge regression same complexity with linear least square
from sklearn.linear_model import Ridge

reg = Ridge(alpha=0.5)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, 0.1, 1])
print(reg.coef_, reg.intercept_)

# %% Ridge CV Generalized Cross-Validation
from sklearn.linear_model import RidgeCV
reg = RidgeCV(alphas=[0.15, 0.1, 0.3, 1.0])
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
print(reg.alpha_, reg.coef_)

# %% the influence of Ridge Coeff to weight
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)

coefs = []
for a in alphas:
    rg = Ridge(alpha=a, fit_intercept=False)
    rg.fit(X, y)
    coefs.append(rg.coef_)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficents as a function of the regularization')
plt.axis('tight')
plt.show()

# %% LASSO
from sklearn.linear_model import Lasso
reg = Lasso(alpha=0.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
reg.predict([[1, 1]])

# %%
