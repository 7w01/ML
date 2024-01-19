import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn import linear_model

# normalization
scale = False

# input data
data = np.genfromtxt('data_.csv', delimiter=',')
x_data = data[:, :-1]
y_data = data[:, -1]

def plot():
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    for i in range(len(y_data)):
        if y_data[i] == 0:
            x0.append(x_data[i][0])
            y0.append(x_data[i][1])
        else:
            x1.append(x_data[i][0])
            y1.append(x_data[i][1])
    #graph
    scatter0 = plt.scatter(x0, y0, c='r', marker='x')
    scatter1 = plt.scatter(x1, y1, c='b', marker='o')

    plt.legend(handles=[scatter0, scatter1], labels=['label0', 'labels1'], loc='best')


model = linear_model.LogisticRegression()
model.fit(x_data, y_data)

plot()
x_test = np.array([[1], [4]])
y_test = (-model.intercept_ - x_test * model.coef_[0][0]) / model.coef_[0][1]
plt.plot(x_test, y_test, 'k')
plt.show()

