import numpy as np
import matplotlib.pyplot as plt

#input data
data = np.genfromtxt("data.csv", delimiter=',')
x_data = data[:, 0]
y_data = data[:, 2]
plt.scatter(x_data, y_data)
plt.show()

#learning rate
lr = 0.0000001
#intercept
b = 0
#slope
k = 0
#epochs
epochs = 50

#Least Square
def cost_function(b, k, x_data, y_data):
    total_error = 0
    for i in range(0, len(x_data)):
        total_error += (y_data[i] - (k * x_data[i] + b)) ** 2
    return total_error / (2 * float(len(x_data)))


def gradient_descent(x_data, y_data, b, k, lr, epochs):
    m = float(len(x_data))
    for i in range(epochs):
        b_temp = 0
        k_temp = 0
        for j in range(0, len(x_data)):
            b_temp += -(1 / m) * (y_data[j] - (k * x_data[j] + b))
            k_temp += -(1 / m) * (y_data[j] - (k * x_data[j] + b)) * x_data[j]
        b = b - (lr * b_temp)
        k = k - (lr * k_temp)
    return b, k


print('Starting b = {0}, k = {1}, error = {2}'.format(b, k, cost_function(b, k, x_data, y_data)))
print("Running...")
b, k = gradient_descent(x_data, y_data, b, k, lr, epochs)
print("After {0} iterations b = {1}, k = {2}, error = {3}".format(epochs, b, k, cost_function(b, k, x_data, y_data)))

#graph
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, k * x_data + b, 'r')
plt.show()
