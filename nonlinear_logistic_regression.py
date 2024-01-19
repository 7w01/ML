import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_gaussian_quantiles

# ganerate data
x_data, y_data = make_gaussian_quantiles(n_samples=500, n_features=2, n_classes=2)

# polynomial conversion
poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x_data)

# modeling
model = LogisticRegression()
model.fit(x_poly, y_data)

xx, yy = np.meshgrid(np.arange(x_data[:, 0].min() - 1, x_data[:, 0].max() + 1), np.arange(x_data[:, 1].min() - 1, x_data[:, 1].max() + 1))
z = model.predict(poly.fit_transform(np.c_[xx.ravel(), yy.ravel()]))
z = z.reshape(xx.shape)

# graph
plt.contourf(xx, yy, z)
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()