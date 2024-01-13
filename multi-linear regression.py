import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# input data
data = np.genfromtxt(r'data.csv', delimiter=',')
# print(data)
x_data = data[:, :-1]
y_data = data[:, -1]
print(x_data)
print(y_data)

# learning rate
lr = 0.0000001
# parameters
theta = np.zeros((3,), dtype=np.float64)
# epochs
epochs = 50

# Least Squre
def cost_function(theta, x_data, y_data):
    total_error = 0
    for i in range(len(x_data)):
        temp = y_data[i]
        for j in range(len(theta)):
            temp -= theta[j] * x_data[i, j]
        temp = temp ** 2
    return total_error / (2 * float(len(x_data)))


def gradient_descent(x_data, y_data, theta, lr, epochs):
    m = float(len(x_data))
    for i in range(epochs):
        theta_temp = np.zeros(np.shape(theta), dtype=np.float64)
        for j in range(len(x_data)):
            theta_temp[0] += -(1 / m) * (y_data[j] - (theta[0] + theta[1] * x_data[j, 0] + theta[2] * x_data[j, 1]))
            theta_temp[1] += -(1 / m) * x_data[j, 0] * (y_data[j] - (theta[0] + theta[1] * x_data[j, 0] + theta[2] * x_data[j, 1]))
            theta_temp[2] += -(1 / m) * x_data[j, 1] * (y_data[j] - (theta[0] + theta[1] * x_data[j, 0] + theta[2] * x_data[j, 1]))
        for k in range(len(theta)):
            theta[k] -= lr * theta_temp[k]
    return theta


print("Running...")
theta = gradient_descent(x_data, y_data, theta, lr, epochs)

# 3d graph
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(x_data[:, 0], x_data[:, 1], y_data, c='r', marker='o', s=100)
x0 = x_data[:, 0]
x1 = x_data[:, 1]

# generate grid matrix
x0, x1 = np.meshgrid(x0, x1)
z = theta[0] + x0 * theta[1] + x1 * theta[2]
ax.plot_surface(x0, x1, z)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()