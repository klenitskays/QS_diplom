import pandas as pd
import numpy as np

# Укажите путь к исходному файлу
input_file = 'startdata.xlsx'

# Чтение исходного файла
df = pd.read_excel(input_file)

# 1. Выручка без НДС = Количество позиций * Цена за единицу товара * (1 - ставка НДС)
df['Выручка без НДС'] = df['OrderItemQuantity'] * df['PerUnitPrice'] * (1 - 0.2)

# 2. Выручка = Количество позиций * Цена за единицу товара
df['Выручка'] = df['OrderItemQuantity'] * df['PerUnitPrice']

# 3. Валовая прибыль = (Количество позиций * Цена за единицу товара) - (Количество позиций * Стандартная стоимость продукта)
df['Валовая прибыль'] = (df['OrderItemQuantity'] * df['PerUnitPrice']) - (df['OrderItemQuantity'] * df['ProductStandardCost'])

# 4. План выручки = ОКРУГЛ(Выручка без НДС * (случайное число от 0 до 2), 2 знака после запятой)
df['План выручки'] = np.round(df['Выручка без НДС'] * (np.random.rand() * 2), 2)

# 5. План валовой прибыли = ОКРУГЛ(Валовая прибыль * (случайное число от 0 до 2), 2 знака после запятой)
df['План валовой прибыли'] = np.round(df['Валовая прибыль'] * (np.random.rand() * 2), 2)

# 6. План продаж = ОКРУГЛ(Количество позиций * (случайное число от 0 до 1.5), 0)
df['План продаж'] = np.round(df['OrderItemQuantity'] * (np.random.rand() * 1.5), 0)

# 7. Общая стоимость заказа = Количество позиций * Цена за единицу товара
df['Общая стоимость заказа'] = df['OrderItemQuantity'] * df['PerUnitPrice']

# 12. Стаж работы сотрудника: переводим дату найма в формат datetime
df['Дата найма сотрудника'] = pd.to_datetime(df['EmployeeHireDate'], format='%Y-%m-%d')
df['Стаж работы (дни)'] = (pd.to_datetime("today") - df['Дата найма сотрудника']).dt.days
df['Стаж работы (годы)'] = df['Стаж работы (дни)'] // 365

# 13. Остаток кредитного лимита = Кредитный лимит клиента - Общая сумма расходов клиента
df['Остаток кредитного лимита'] = df['CustomerCreditLimit'] - df['Общая стоимость заказа']

# Проверка на пропущенные значения и их обработка
df = df.dropna(subset=['Стаж работы (годы)', 'ProductStandardCost', 'OrderItemQuantity', 'PerUnitPrice', 'Выручка', 'Валовая прибыль'])

# Сохраняем результат в новый Excel-файл
output_file = 'startdata_with_new_columns.xlsx'
df.to_excel(output_file, index=False)

print(f"\nДанные с новыми столбцами сохранены в {output_file}")
