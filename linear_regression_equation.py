import numpy as np
import matplotlib.pyplot as plt

# input data
data = np.genfromtxt('data.csv', delimiter=',')
x_data = data[:, 0, np.newaxis]# because x and y will be transformed into a matrix
y_data = data[:, -1, np.newaxis]

# add intercepts
X_data = np.concatenate((np.ones((len(x_data), 1)), x_data), axis=1)# concatenate x with ones


def weights(x_arr, y_arr):
    X = np.mat(x_arr)
    Y = np.mat(y_arr)
    XTX = X.T * X
    if np.linalg.det(XTX) == 0:
        print("XTX is single")
        return
    w = XTX.I * X.T * Y
    return w


w = weights(X_data, y_data)
print("The coefficents Ws are")
print(w)

# graph
x_test = np.array([[1000], [4500]])
y_test = w[0] + x_test * w[1]# notice the order
plt.plot(x_data, y_data, 'b.')
plt.plot(x_test, y_test, 'r')
plt.show()