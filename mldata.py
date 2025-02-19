import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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

# Модель линейной регрессии для выручки
revenue_model = LinearRegression()
revenue_model.fit(X_train, y_train_revenue)

# Прогнозирование выручки
df['Predicted_Выручка'] = revenue_model.predict(scaler.transform(df[features]))

# Модель линейной регрессии для валовой прибыли
profit_model = LinearRegression()
profit_model.fit(X_train, y_train_profit)

# Прогнозирование валовой прибыли
df['Predicted_Валовая_прибыль'] = profit_model.predict(scaler.transform(df[features]))

# Классификация клиентов по кредитному лимиту (анализ риска)
df['HighCreditRisk'] = df['CustomerCreditLimit'].apply(lambda x: 1 if x < 10000 else 0)

# Признаки для классификации
X_class = df[['Стаж работы (годы)', 'PerUnitPrice', 'OrderItemQuantity', 'ProductStandardCost']]
y_class = df['HighCreditRisk']

# Масштабирование признаков для классификации
X_class_scaled = scaler.fit_transform(X_class)

# Разделение на тренировочную и тестовую выборки
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class_scaled, y_class, test_size=0.2, random_state=42)

# Модель случайного леса для классификации
clf = RandomForestClassifier()
clf.fit(X_train_class, y_train_class)

# Прогнозирование риска
df['Predicted_HighCreditRisk'] = clf.predict(scaler.transform(df[['Стаж работы (годы)', 'PerUnitPrice', 'OrderItemQuantity', 'ProductStandardCost']]))

# Сохраняем результат в новый Excel-файл
output_file = 'final_data_with_predictions.xlsx'
df.to_excel(output_file, index=False)

print(f"\nДанные с прогнозами сохранены в {output_file}")
