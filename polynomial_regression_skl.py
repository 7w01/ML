import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# input data
data = np.genfromtxt('poly.csv', delimiter=',')
x_data = data[:, 0, np.newaxis]
y_data = data[:, 1]

# polynomial conversion
poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x_data)
# [1,2,3] -> [[1,1,1],[1,2,3],[1,4,9],[1,16,27]]
print(x_data)
print(x_poly)

# linear model
model = LinearRegression()
model.fit(x_poly, y_data)

#graph
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, model.predict(poly.fit_transform(x_data)), 'r')
plt.show()
