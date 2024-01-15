import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# input data
data = np.genfromtxt('data1.csv', delimiter=',')
x_data = data[:, 2:]
y_data = data[:, 1]

# ridge coefficents
lambda_test = np.linspace(0.001, 1)# cross-validation

# modeling
model = linear_model.RidgeCV(alphas=lambda_test, store_cv_values=True)
model.fit(x_data, y_data)
print(model.alpha_)# alpha_: best ridge coefficent

# graph
plt.plot(lambda_test, model.cv_values_.mean(axis=0)) # cv_values: matrix of loss values
plt.plot(model.alpha_, min(model.cv_values_.mean(axis=0)), 'ro')
plt.show()