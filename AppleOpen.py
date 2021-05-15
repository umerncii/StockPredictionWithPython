
#Predicting Apple open stock prices by using LSTM
#Importing all my packages.

import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as web
from keras.losses import mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime as dt

#Getting data
company= 'AAPL'
start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)
data = web.DataReader(company, 'yahoo', start, end)

#Getting ready my data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(data['Open'].values.reshape(-1,1))
prediction_days= 60

x_train=[]
y_train=[]

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train= np.array(x_train), np.array(y_train)
x_train= np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs= 25, batch_size=32)
#Here we are testing our model based on existing data
test_start=dt.datetime(2020,1,1)
test_end=dt.datetime.now()
test_data=web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Open'].values
total_dataset = pd.concat((data['Open'], test_data['Open']), axis=0)
model_inputs= total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs= model_inputs.reshape(-1, 1)
model_inputs =scaler.transform(model_inputs)
x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test= np.array(x_test)
x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
plt.plot(actual_prices, color='Blue', label="Actual Prices ")
plt.plot(predicted_prices, color='red', label= " Predicted apple prices")
plt.title("Apple actual and predicted stock prices")
plt.xlabel('Time')
plt.ylabel('Apple open stock share prices')
plt.legend()
plt.show()


apple_quote=web.DataReader('AAPL', data_source='yahoo', start='2015-01-01', end='2021-05-15')
new_df=apple_quote.filter(['Open'])
last_60_days=new_df[-60:].values
last_60_days_scaled=scaler.transform(last_60_days)
X_test=[]
X_test.append(last_60_days_scaled)
X_test=np.array(X_test)
X_test=np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price=model.predict(X_test)
pred_price=scaler.inverse_transform(pred_price)
print(pred_price)#123.5206 (2021-05-15)

apple_quote2=web.DataReader('AAPL', data_source='yahoo', start='2015-01-01', end='2021-05-20')
new_df2=apple_quote2.filter(['Open'])
last_60_days2=new_df2[-60:].values
last_60_days_scaled2=scaler.transform(last_60_days2)
X_test2=[]
X_test2.append(last_60_days_scaled2)
X_test2=np.array(X_test2)
X_test2=np.reshape(X_test2, (X_test2.shape[0], X_test2.shape[1], 1))
second_pred_price=model.predict(X_test2)
second_pred_price=scaler.inverse_transform(second_pred_price)
print(second_pred_price)#124.61107(2021-05-20)

apple_quote3=web.DataReader('AAPL', data_source='yahoo', start='2015-01-01', end='2021-05-25')
new_df3=apple_quote3.filter(['Open'])
last_60_days3=new_df3[-60:].values
last_60_days_scaled3=scaler.transform(last_60_days3)
X_test3=[]
X_test3.append(last_60_days_scaled3)
X_test3=np.array(X_test3)
X_test3=np.reshape(X_test3, (X_test3.shape[0], X_test3.shape[1], 1))
third_pred_price=model.predict(X_test3)
third_pred_price=scaler.inverse_transform(third_pred_price)
print(third_pred_price)#124.642265(2021-05-25)

apple_quote4=web.DataReader('AAPL', data_source='yahoo', start='2015-01-01', end='2021-05-27')
new_df4=apple_quote4.filter(['Open'])
last_60_days4=new_df4[-60:].values
last_60_days_scaled4=scaler.transform(last_60_days4)
X_test4=[]
X_test4.append(last_60_days_scaled4)
X_test4=np.array(X_test4)
X_test4=np.reshape(X_test4, (X_test4.shape[0], X_test4.shape[1], 1))
fourth_pred_price=model.predict(X_test4)
fourth_pred_price=scaler.inverse_transform(fourth_pred_price)
print(fourth_pred_price)#123.00415(2021-05-27)

apple_quote5=web.DataReader('AAPL', data_source='yahoo', start='2015-01-01', end='2021-05-30')
new_df5=apple_quote5.filter(['Open'])
last_60_days5=new_df5[-60:].values
last_60_days_scaled5=scaler.transform(last_60_days5)
X_test5=[]
X_test5.append(last_60_days_scaled5)
X_test5=np.array(X_test5)
X_test5=np.reshape(X_test5, (X_test5.shape[0], X_test5.shape[1], 1))
fifth_pred_price=model.predict(X_test5)
fifth_pred_price=scaler.inverse_transform(fifth_pred_price)
print(fifth_pred_price)#2021-05-30










