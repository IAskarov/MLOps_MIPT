import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import joblib

# Загрузка модели
model = joblib.load("model.pkl")

# Загрузка валидационного датасета
iris = load_iris()
X_val = iris.data
y_val = iris.target

# Предсказание ответов для валидационного датасета
predictions = model.predict(X_val)

# Запись результатов в .csv файл
df = pd.DataFrame({"observation": range(len(predictions)), "predicted_label": predictions})
df.to_csv("predictions.csv", index=False)

# Вычисление метрик и вывод результатов
accuracy = (predictions == y_val).mean()
print("Accuracy:", accuracy)