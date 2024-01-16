import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import classification_report
from mpl_toolkits.mplot3d import Axes3D

# normalization
scale = False

# input data
data = np.genfromtxt('data_.csv', delimiter=',')
x_data = data[:, :-1]
y_data = data[:, -1, np.newaxis]


def plot():
    x0 = []
    y0 = []
    z0 = []
    x1 = []
    y1 = []
    z1 = []
    for i in range(len(y_data)):
        if y_data[i][0] == 0:
            x0.append(x_data[i][0])
            y0.append(x_data[i][1])
            z0.append(x_data[i][2])
        else:
            x1.append(x_data[i][0])
            y1.append(x_data[i][1])
            z1.append(x_data[i][2])
    #graph
    ax = plt.figure().add_subplot(111, projection='3d')
    scatter0 = ax.scatter(x0, y0, z0, c='r', marker='x')
    scatter1 = ax.scatter(x1, y1, z1, c='b', marker='o')

    x0_test = [1, 2, 3]
    x1_test = [1, 2, 3]
    x0_test, x1_test = np.meshgrid(x0_test, x1_test)
    x2_test = (-Ws[0] - x0_test * Ws[1] - x1_test * Ws[2]) / Ws[3]
    ax.plot_surface(x0_test, x1_test, x2_test)

    plt.legend(handles=[scatter0, scatter1], labels=['label0', 'labels1'], loc='best')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def cost_function(xMat,yMat, Ws):
    return 1 / len(y_data) * np.sum(np.multiply(yMat, xMat * Ws) - np.log(1 + np.exp(xMat * Ws)))


def gradient_descent(xArr, yArr):
    if scale == True:
        xArr = preprocessing.scale(xArr)

    xMat = np.mat(xArr)
    yMat = np.mat(yArr)

    lr = 0.01
    epochs = 1000
    cost_list = []
    Ws = np.mat(np.ones((len(xArr[0]), 1)))

    for i in range(epochs):
        Ws -= lr * xMat.T * (sigmoid(xMat * Ws) - yMat) / len(y_data)

        if i % 50 == 0:
            cost_list.append(cost_function(xMat, yMat, Ws))

    return Ws, cost_list


# add intercept
X_data = np.concatenate((np.ones((len(y_data), 1)), x_data), axis=1)

Ws, cost_list = gradient_descent(X_data, y_data)
Ws = np.array(Ws)
Ws = Ws[:, 0]
print('W = ', Ws)
print('cost = ', cost_list)

plot()
plt.show()
