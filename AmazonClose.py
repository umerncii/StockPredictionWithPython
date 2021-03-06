#First Name: Umer

#Last Name: Iqbal

#Student No: x17111854

import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as web
from keras.losses import mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#Here I am setting the path of my Amazon Dataset and storing it in appleData variable
amznData = pd.read_csv(r'D:/4th Year/Final Year Project/Mid-Implementation/NewYork Stock Exchange/AMZN.csv')
#Printing my Apple dataset
print(amznData)

#Let's see numbers of rows and columns in datasets
amznData.shape

#Visualize the Apple closing price history
plt.figure(figsize=(15,7))
plt.title('Amazon closing price')
plt.plot(amznData['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD $', fontsize=16)
plt.show()
#Creating a new dataframe with only our close column
data = amznData.filter(['Close'])
#Let's convert dataframe to numpy array
dataset= data.values
#Getting the number of rows for training our model
training_data_len= math.ceil(len(dataset) * .8)
#Print the value
print(training_data_len)
#Scale our data
scaler= MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
#Printing our values
print(scaled_data)

#Let's create scaled training dataset
train_data= scaled_data[0:training_data_len, :]
#Let's split our data into x_train and y_train datastes
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60: i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()

#Here I am converting x_train and y_train dataset to numpy Array
x_train, y_train=np.array(x_train), np.array(y_train)
#Reshaping our dataset to 3Dimensional
x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

#Building LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False, ))
model.add(Dense(25))
model.add(Dense(1))

#Let's compile our model here
model.compile(optimizer='adam', loss='mean_squared_error')
#Train the model
history=model.fit(
    x_train,
    y_train,
    batch_size=1,
    epochs=1
)
history.history

#Here I am creating new array containing scaled values from index 2134
test_data= scaled_data[training_data_len - 60: , :]
#Creating dataset x_test and y_test
x_test= []
y_test=dataset[training_data_len: , :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60: i, 0])
#Converting the test data into numpy array
x_test=np.array(x_test)
#Reshaping our x_test data from 2D to 3D
x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
#Models predicted price values
predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)
#Finding root mean square value
rmse=np.sqrt(np.mean(predictions-y_test)**2)
print(rmse)

#plot data
train=data[:training_data_len]
valid=data[training_data_len:]
valid['Predictions']=predictions
#Visualize the data
plt.figure(figsize=(12,5))
plt.title('Amazon Closing Prediction')
plt.xlabel('Day 2010-2020', fontsize=18)
plt.ylabel('closing price in $', fontsize=16)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
#Displaying valid and predicted price
print(valid)

amzn_quote=web.DataReader('AMZN', data_source='yahoo', start='2015-01-01', end='2021-04-05')
new_df=amzn_quote.filter(['Close'])
last_60_days=new_df[-60:].values
last_60_days_scaled=scaler.transform(last_60_days)
X_test=[]
X_test.append(last_60_days_scaled)
X_test=np.array(X_test)
X_test=np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price=model.predict(X_test)
pred_price=scaler.inverse_transform(pred_price)
print(pred_price)#3001.4458 for (2021-04-05)

amzn_quot=web.DataReader('AMZN', data_source='yahoo', start='2015-01-01', end='2021-04-07')
new_d=amzn_quot.filter(['Close'])
last_60_day=new_d[-60:].values
last_60_days_scale=scaler.transform(last_60_day)
X_tes=[]
X_tes.append(last_60_days_scale)
X_tes=np.array(X_tes)
X_tes=np.reshape(X_tes, (X_tes.shape[0], X_tes.shape[1], 1))
pred_pric=model.predict(X_tes)
pred_pric=scaler.inverse_transform(pred_pric)
print(pred_pric)#3167.9893(2021-04-07)

amzn_quote3=web.DataReader('AMZN', data_source='yahoo', start='2015-01-01', end='2021-04-14')
new_df3=amzn_quote3.filter(['Close'])
last_60_days3=new_df3[-60:].values
last_60_days_scaled3=scaler.transform(last_60_days3)
X_test3=[]
X_test3.append(last_60_days_scaled3)
X_test3=np.array(X_test3)
X_test3=np.reshape(X_test3, (X_test3.shape[0], X_test3.shape[1], 1))
third_pred_price=model.predict(X_test3)
third_pred_price=scaler.inverse_transform(third_pred_price)
print(third_pred_price)#2926.3347(2021-04-14)

amzn_quote4=web.DataReader('AMZN', data_source='yahoo', start='2015-01-01', end='2021-01-10')
new_df4=amzn_quote4.filter(['Close'])
last_60_days4=new_df4[-60:].values
last_60_days_scaled4=scaler.transform(last_60_days4)
X_test4=[]
X_test4.append(last_60_days_scaled4)
X_test4=np.array(X_test4)
X_test4=np.reshape(X_test4, (X_test4.shape[0], X_test4.shape[1], 1))
fourth_pred_price=model.predict(X_test4)
fourth_pred_price=scaler.inverse_transform(fourth_pred_price)
print(fourth_pred_price)#2850.9338(2021-01-10)

amzn_quote5=web.DataReader('AMZN', data_source='yahoo', start='2015-01-01', end='2021-01-15')
new_df5=amzn_quote5.filter(['Close'])
last_60_days5=new_df5[-60:].values
last_60_days_scaled5=scaler.transform(last_60_days5)
X_test5=[]
X_test5.append(last_60_days_scaled5)
X_test5=np.array(X_test5)
X_test5=np.reshape(X_test5, (X_test5.shape[0], X_test5.shape[1], 1))
fifth_pred_price=model.predict(X_test5)
fifth_pred_price=scaler.inverse_transform(fifth_pred_price)
print(fifth_pred_price)#2810.199(2021-01-15)

