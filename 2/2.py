import pandas as pd

# 1. Создаем объект Series
series = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])

# 2. Обращение по явному индексу для получения значения 4
value_by_explicit_index = series['d']

# 3. Обращение по неявному индексу для получения значения 2
value_by_implicit_index = series[1]  # Индекс 1 соответствует значению 2

# 4. Добавление нового элемента в серию
series['f'] = 6

# 5. Получение значений 3, 4, 5 с помощью операции среза
sliced_values = series['c':'e']

# 6. Создаем объект DataFrame
data = [[1, 2], [5, 3], [3.7, 4.8]]
df = pd.DataFrame(data, columns=['col1', 'col2'])

# 7. Получение элемента 3.7 с помощью операции индексации
element_37 = df['col1'][2]

# 8. Изменение элемента 3 на 9
df.loc[1, 'col1'] = 9

# 9. Получение строк с индексами 1 и 2 с помощью операции среза
sliced_rows = df.loc[1:2]

# 10. Добавление столбца col3, значения которого – результат поэлементного перемножения col1 и col2
df['col3'] = df['col1'] * df['col2']

# Вывод результатов
print("1. Серия:")
print(series)
print("\n2. Значение 4 по явному индексу:", value_by_explicit_index)
print("3. Значение 2 по неявному индексу:", value_by_implicit_index)
print("4. Серия после добавления нового элемента:")
print(series)
print("\n5. Значения 3, 4, 5 по срезу:")
print(sliced_values)
print("\n6. DataFrame:")
print(df)
print("\n7. Элемент 3.7:")
print(element_37)
print("\n8. DataFrame после изменения элемента 3 на 9:")
print(df)
print("\n9. Строки с индексами 1 и 2 по срезу:")
print(sliced_rows)
print("\n10. DataFrame с добавленным результатом поэлементного перемножения col1 и col2:")
print(df)
