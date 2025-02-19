import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import numpy as np

# Укажите путь к обновленному файлу
input_file = 'startdata_with_new_columns.xlsx'

# Чтение обновленного файла
df = pd.read_excel(input_file)

# --- Прогнозирование с использованием машинного обучения ---

# Прогнозирование выручки с помощью линейной регрессии
features = ['Стаж работы (годы)', 'ProductStandardCost', 'OrderItemQuantity', 'PerUnitPrice']
X = df[features]
y_revenue = df['Выручка']
y_profit = df['Валовая прибыль']

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train_revenue, y_test_revenue = train_test_split(X_scaled, y_revenue, test_size=0.2, random_state=42)
X_train, X_test, y_train_profit, y_test_profit = train_test_split(X_scaled, y_profit, test_size=0.2, random_state=42)

# Модели для прогнозирования выручки и валовой прибыли
models = {
    'Linear Regression': LinearRegression(),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'SVM': SVR(kernel='rbf'),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'XGBoost': XGBRegressor(random_state=42)
}

# Прогнозирование выручки и валовой прибыли с использованием всех моделей
for model_name, model in models.items():
    # Прогнозирование для выручки
    model.fit(X_train, y_train_revenue)
    df[f'Predicted_Выручка_{model_name}'] = model.predict(X_scaled)
    
    # Прогнозирование для валовой прибыли
    model.fit(X_train, y_train_profit)
    df[f'Predicted_Валовая_прибыль_{model_name}'] = model.predict(X_scaled)

# --- Визуализация ---

# Сравнение предсказаний
plt.figure(figsize=(14, 8))

# График для выручки
plt.subplot(2, 1, 1)
for model_name in models:
    plt.plot(df['Выручка'], df[f'Predicted_Выручка_{model_name}'], 'o', label=model_name, alpha=0.7)
plt.title('Comparison of Predicted vs Actual Revenue')
plt.xlabel('Actual Revenue')
plt.ylabel('Predicted Revenue')
plt.legend()

# График для валовой прибыли
plt.subplot(2, 1, 2)
for model_name in models:
    plt.plot(df['Валовая прибыль'], df[f'Predicted_Валовая_прибыль_{model_name}'], 'o', label=model_name, alpha=0.7)
plt.title('Comparison of Predicted vs Actual Gross Profit')
plt.xlabel('Actual Gross Profit')
plt.ylabel('Predicted Gross Profit')
plt.legend()

plt.tight_layout()
plt.show()

# Сохраняем результат в новый Excel-файл
output_file = 'final_data_with_predictions.xlsx'
df.to_excel(output_file, index=False)

print(f"\nДанные с прогнозами сохранены в {output_file}")
