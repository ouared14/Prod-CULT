import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

data = pd.read_csv('olive_production_dataset.csv')
data['Date'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['Month'].astype(str) + '-01')
data.set_index('Date', inplace=True)
data.fillna(method='ffill', inplace=True)
production_data = data['Target (Production)']

def check_stationarity(series):
    result = adfuller(series)
    if result[1] > 0.05:
        return series.diff().dropna()
    return series

production_data_diff = check_stationarity(production_data)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(production_data_diff, lags=30, ax=plt.gca())
plt.subplot(122)
plot_pacf(production_data_diff, lags=30, ax=plt.gca())
plt.show()

model_1 = ARIMA(production_data, order=(1, 1, 1))
model_1_fit = model_1.fit()

exogenous_data = data[['Tmin', 'Tmax', 'Precipitation']]
model_2 = ARIMA(production_data, exog=exogenous_data, order=(1, 1, 1))
model_2_fit = model_2.fit()

model_1_fit.plot_diagnostics(figsize=(10, 6))
plt.show()

model_2_fit.plot_diagnostics(figsize=(10, 6))
plt.show()

forecast_1 = model_1_fit.forecast(steps=10)
forecast_2 = model_2_fit.forecast(steps=10, exog=exogenous_data.iloc[-10:])

plt.figure(figsize=(10, 6))
plt.plot(production_data, label='Original Production Data')
plt.plot(forecast_1, label='ARIMA(1,1,1) Forecast', linestyle='--')
plt.plot(forecast_2, label='ARIMA with Exogenous Variables Forecast', linestyle='--')
plt.legend()
plt.show()
