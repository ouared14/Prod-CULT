import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

data = pd.read_csv('olive_production_dataset.csv')
data['Date'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['Month'].astype(str) + '-01')
data.set_index('Date', inplace=True)
data.fillna(method='ffill', inplace=True)
production_data = data['Target (Production)'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
production_data_scaled = scaler.fit_transform(production_data)

def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X, Y = create_dataset(production_data_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, Y, epochs=100, batch_size=32)

test_size = 10
test_data = production_data_scaled[-(test_size + time_step):]
X_test, Y_test = create_dataset(test_data, time_step)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

train = data[:len(data) - test_size]
valid = data[-test_size:]
valid['Predictions'] = predictions

plt.figure(figsize=(10, 6))
plt.plot(train['Target (Production)'])
plt.plot(valid[['Target (Production)', 'Predictions']])
plt.show()
