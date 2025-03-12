import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, Add, MultiHeadAttention

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(max_len, d_model)

    def get_angles(self, position, d_model):
        angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model // 2) // 2)) / np.float32(d_model))
        return position * angle_rates

    def positional_encoding(self, max_len, d_model):
        angles = self.get_angles(np.arange(max_len)[:, np.newaxis], d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        pos_encoding = angles[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

data = pd.read_csv('olive_production_dataset.csv')
data['Date'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['Month'].astype(str) + '-01')
data.set_index('Date', inplace=True)
data.fillna(method='ffill', inplace=True)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Target (Production)']])

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 12
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

d_model = 64
num_heads = 4

inputs = Input(shape=(time_step, 1))
x = Dense(d_model)(inputs)
x = PositionalEncoding(d_model)(x)
x = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = LayerNormalization()(x)
x = Add()([x, inputs])  # Residual connection
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

plt.figure(figsize=(10, 6))
plt.plot(scaler.inverse_transform(scaled_data), label='Original Data')
plt.plot(np.arange(time_step, len(train_predict) + time_step), train_predict, label='Train Prediction')
plt.plot(np.arange(len(train_predict) + (time_step * 2), len(train_predict) + (time_step * 2) + len(test_predict)), test_predict, label='Test Prediction')
plt.legend()
plt.show()
