import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Зчитування даних з файлу task51.txt
data = pd.read_csv('task51.txt', delimiter='\t', parse_dates=['Дата'], dayfirst=True)
data.set_index('Дата', inplace=True)

# Виведення перших рядків даних
print("Перші рядки даних з task51.txt:")
print(data.head())

# Збереження даних у файл Lab5.xlsx та Lab5.txt
data.to_excel('Lab5.xlsx')
data.to_csv('Lab5.txt', sep='\t')

# Зчитування тестових даних з файлу test51.txt
test_data = pd.read_csv('test51.txt', delimiter='\t', parse_dates=['Дата'], dayfirst=True)
test_data.set_index('Дата', inplace=True)

# Збереження тестових даних у файл Labtest5.txt
test_data.to_csv('Labtest5.txt', sep='\t')

# Візуалізація тренду
plt.figure(figsize=(12, 6))
plt.plot(data['Курс 100 EUR/UAH'], label='Курс 100 EUR/UAH')
plt.title('Тренд курсу 100 EUR/UAH')
plt.xlabel('Дата')
plt.ylabel('Курс')
plt.legend()
plt.show()

# Автокореляція
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data['Курс 100 EUR/UAH'], lags=30)
plt.title('Автокореляція курсу 100 EUR/UAH')
plt.show()

# Побудова серії трендів методом ковзаючого вікна
window_size = 7
data['Ковзаюче середнє'] = data['Курс 100 EUR/UAH'].rolling(window=window_size).mean()

# Візуалізація ковзаючого середнього
plt.figure(figsize=(12, 6))
plt.plot(data['Курс 100 EUR/UAH'], label='Курс')
plt.plot(data['Ковзаюче середнє'], label=f'Ковзаюче середнє (вікно={window_size})')
plt.title('Ковзаюче середнє курсу 100 EUR/UAH')
plt.xlabel('Дата')
plt.ylabel('Курс')
plt.legend()
plt.show()

# Нормалізація даних
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Курс 100 EUR/UAH']])

# Створення послідовностей даних для навчання
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10
X, y = create_sequences(scaled_data, seq_length)

# Розділення на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Побудова моделі LSTM
model_nn = Sequential()
model_nn.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model_nn.add(Dense(1))
model_nn.compile(optimizer='adam', loss='mse')

# Навчання моделі LSTM
model_nn.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Прогнозування за допомогою LSTM
predictions_nn = model_nn.predict(X_test)
predictions_nn = scaler.inverse_transform(predictions_nn)

# Візуалізація результатів LSTM
plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(y_test):], scaler.inverse_transform(y_test), label='Фактичний курс')
plt.plot(data.index[-len(y_test):], predictions_nn, label='Прогнозований курс (нейронна мережа)')
plt.title('Прогноз курсу 100 EUR/UAH за допомогою нейронної мережі')
plt.xlabel('Дата')
plt.ylabel('Курс')
plt.legend()
plt.show()

# Побудова лінійної регресії
X = np.arange(len(data)).reshape(-1, 1)
y = data['Курс 100 EUR/UAH'].values

# Розділення на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Побудова моделі лінійної регресії
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Прогнозування за допомогою лінійної регресії
predictions_lr = model_lr.predict(X_test)

# Візуалізація результатів лінійної регресії
plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(y_test):], y_test, label='Фактичний курс')
plt.plot(data.index[-len(y_test):], predictions_lr, label='Прогнозований курс (лінійна регресія)')
plt.title('Прогноз курсу 100 EUR/UAH за допомогою лінійної регресії')
plt.xlabel('Дата')
plt.ylabel('Курс')
plt.legend()
plt.show()

# Порівняння результатів
mse_nn = mean_squared_error(y_test, predictions_nn)
mse_lr = mean_squared_error(y_test, predictions_lr)

print(f'MSE нейронної мережі: {mse_nn}')
print(f'MSE лінійної регресії: {mse_lr}')