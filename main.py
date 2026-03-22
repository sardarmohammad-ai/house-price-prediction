import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([[50], [60], [70], [80], [90]])
y = np.array([100000, 120000, 140000, 160000, 180000])

model = LinearRegression()
model.fit(x,y)

size = 100
predicted_price = model.predict([[size]])

print(f"Predicted price for {size}m²  house: {predicted_price[0]:.2f}")

plt.scatter(x,y, color='blue')
plt.plot(x, model.predict(x), color='red')
plt.xlabel("House Size")
plt.ylabel("Price")
plt.title("House Price Prediction")
plt.show()
