from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

#input data
data = np.genfromtxt('data.csv', delimiter=',')
#format conversion
x_data = data[:, 0, np.newaxis]
y_data = data[:, 2, np.newaxis]
plt.scatter(x_data, y_data)
plt.show()

#modeling
model = LinearRegression()
model.fit(x_data, y_data)

#graph
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, model.predict(x_data), 'r')
plt.show()