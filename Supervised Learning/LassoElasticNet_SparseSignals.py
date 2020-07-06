"""
Estimate Lasso and Elastic-Net regression models on a manually generated sparse
signal corrupted with an additive noise.
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
np.random.seed(42)

n_samples, n_features = 50, 100
X = np.random.randn(n_samples, n_features)

idx = np.arange(n_features)
coef = (-1)**idx * np.exp(-idx / 10)
coef[10:] = 0
y = np.dot(X, ceof)

y += 0.01 * np.random.normal(size=n_samples)
n_samples = X.shape[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#  %% Lasso
from sklearn.linear_model import Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(lasso)
print('r^2 on test data : %f' % r2_score_lasso)

# %% ElasticNet
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print(enet)
print('r^2 on test data : %f' % r2_score_enet)

m, s, _ = plt.stem(np.where(enet.coef_)[0],
                   enet.coef_[enet.coef_ != 0],
                   markerfmt='x',
                   label='Elastic net coefficents',
                   use_line_collection=True)
plt.setp([m, s], color='#2ca02c')

m, s, _ = plt.stem(np.where(lasso.coef_)[0],
                   lasso.coef_[lasso.coef_ != 0],
                   markerfmt='x',
                   label='Lasso coefficents',
                   use_line_collection=True)
plt.setp([m, s], color='#ff7f0e')

m, s, _ = plt.stem(np.where(coef)[0],
                   lasso.coef_[coef != 0],
                   markerfmt='bx',
                   label='True coefficents',
                   use_line_collection=True)

plt.legend(loc='best')
plt.title('Lasso $R^2$: %.3f, Elastic Net $R^2$: %.3f' %
          (r2_score_lasso, r2_score_enet))
plt.show()

# %%
