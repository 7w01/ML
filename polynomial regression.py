import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = np.genfromtxt('data_.csv', delimiter=',')
x_data = data[:, 0, np.newaxis]
y_data = data[:, 1, np.newaxis]

poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x_data)

model = LinearRegression()
model.fit(x_poly, y_data)

plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, model.predict(poly.fit_transform(x_data)), 'r')
plt.show()
