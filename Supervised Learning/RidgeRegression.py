"""Linear Regression Demo
Author: X.F.Zhou
Email: xfzhou233@gmail.com
"""
# %% import the packages
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# %% load the data and split test set
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X = diabetes_X[:, np.newaxis, 2]
X_train, X_test, y_train, y_test = train_test_split(diabetes_X,
                                                    diabetes_y,
                                                    test_size=0.1)

# %% plot the dataset
plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test)
plt.legend(['Train', 'Test'])

# %% fit the model
regr = RidgeCV(np.logspace(-100, 100))
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
print('Coeffiients:\n', regr.coef_)
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))

# %% plot the outputs
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.show()

# %%
