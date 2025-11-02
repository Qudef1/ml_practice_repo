import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

data = fetch_california_housing()



X = data.data[:, 2:3]  # Берём только признак 'RM' (число комнат)
y = data.target

# 2. Обучение модели
model = LinearRegression()
model.fit(X, y)

# 3. Визуализация
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Среднее число комнат')
plt.ylabel('Цена дома')
plt.show()