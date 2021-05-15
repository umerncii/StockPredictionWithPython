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
company= 'AMZN'
start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)
data = web.DataReader(company, 'yahoo', start, end)

#Getting ready my data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(data['Low'].values.reshape(-1,1))
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
actual_prices = test_data['Low'].values

total_dataset = pd.concat((data['Low'], test_data['Low']), axis=0)
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

plt.plot(actual_prices, color='yellow', label="Actual Prices ")
plt.plot(predicted_prices, color='red', label= " Predicted amazon prices")
plt.title("Amazon actual and predicted stock prices")
plt.xlabel('Time')
plt.ylabel('Amazon low stock share prices')
plt.legend()
plt.show()

#Predicting Future/Next day
#real_data = [model_inputs[len(model_inputs) + 1, 0]]
#real_data = np.array(real_data)
#real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

#prediction =model.predict(real_data)
#prediction = scaler.inverse_transform(prediction)
#print(prediction)


amzn_quote=web.DataReader('AMZN', data_source='yahoo', start='2015-01-01', end='2021-05-05')
new_df=amzn_quote.filter(['Low'])
last_60_days=new_df[-60:].values
last_60_days_scaled=scaler.transform(last_60_days)
X_test=[]
X_test.append(last_60_days_scaled)
X_test=np.array(X_test)
X_test=np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price=model.predict(X_test)
pred_price=scaler.inverse_transform(pred_price)
print(pred_price)

amzn_quot=web.DataReader('AMZN', data_source='yahoo', start='2015-01-01', end='2021-05-06')
new_d=amzn_quot.filter(['Low'])
last_60_day=new_d[-60:].values
last_60_days_scale=scaler.transform(last_60_day)
X_tes=[]
X_tes.append(last_60_days_scale)
X_tes=np.array(X_tes)
X_tes=np.reshape(X_tes, (X_tes.shape[0], X_tes.shape[1], 1))
pred_pric=model.predict(X_tes)
pred_pric=scaler.inverse_transform(pred_pric)
print(pred_pric)

amzn_quote3=web.DataReader('AMZN', data_source='yahoo', start='2015-01-01', end='2021-05-07')
new_df3=amzn_quote3.filter(['Low'])
last_60_days3=new_df3[-60:].values
last_60_days_scaled3=scaler.transform(last_60_days3)
X_test3=[]
X_test3.append(last_60_days_scaled3)
X_test3=np.array(X_test3)
X_test3=np.reshape(X_test3, (X_test3.shape[0], X_test3.shape[1], 1))
third_pred_price=model.predict(X_test3)
third_pred_price=scaler.inverse_transform(third_pred_price)
print(third_pred_price)

amzn_quote4=web.DataReader('AMZN', data_source='yahoo', start='2015-01-01', end='2021-05-08')
new_df4=amzn_quote4.filter(['Low'])
last_60_days4=new_df4[-60:].values
last_60_days_scaled4=scaler.transform(last_60_days4)
X_test4=[]
X_test4.append(last_60_days_scaled4)
X_test4=np.array(X_test4)
X_test4=np.reshape(X_test4, (X_test4.shape[0], X_test4.shape[1], 1))
fourth_pred_price=model.predict(X_test4)
fourth_pred_price=scaler.inverse_transform(fourth_pred_price)
print(fourth_pred_price)

amzn_quote5=web.DataReader('AMZN', data_source='yahoo', start='2015-01-01', end='2021-05-09')
new_df5=amzn_quote5.filter(['Low'])
last_60_days5=new_df5[-60:].values
last_60_days_scaled5=scaler.transform(last_60_days5)
X_test5=[]
X_test5.append(last_60_days_scaled5)
X_test5=np.array(X_test5)
X_test5=np.reshape(X_test5, (X_test5.shape[0], X_test5.shape[1], 1))
fifth_pred_price=model.predict(X_test5)
fifth_pred_price=scaler.inverse_transform(fifth_pred_price)
print(fifth_pred_price)

