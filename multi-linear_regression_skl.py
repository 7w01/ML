import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D

# input data
data = np.genfromtxt(r'data.csv', delimiter=',')
x_data = data[:, :-1]
y_data = data[:, -1]

# modeling
model = linear_model.LinearRegression()
model.fit(x_data, y_data)

# 3d graph
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(x_data[:, 0], x_data[:, 1], y_data, c='r', s=100)
x0 = x_data[:, 0]
x1 = x_data[:, 1]

# generate grid matrix
x0, x1 = np.meshgrid(x0, x1)
z = model.intercept_ + x0 * model.coef_[0] + x1 * model.coef_[1]
ax.plot_surface(x0, x1, z)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()