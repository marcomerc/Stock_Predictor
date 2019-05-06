# -*- coding: utf-8 -*-
#data processing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #for data sets


#importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_train.csv')
training_set = dataset_train.iloc[:,1:2].values#values for numpy array #all the rows and look at the csv 1 to 2 but its only the colomn 1 
# we have to normalized the features
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1)) # the range we want to scare the features.
training_set_scaled  = sc.fit_transform(training_set)

# creating a data structure with 60 timesteps and 1 output 
x_train = []
y_train = []

for i in range(60,1258):   # doesnt include 1258 index  it contains the 60 previous days from when we try to predict
    x_train.append(training_set_scaled[i-60:i,0]) #zero colomn and 60 rows for each day 
    y_train.append(training_set_scaled[i,0]) # we need the price on that day  for y as an ouput 
    
x_train, y_train = np.array(x_train), np.array(y_train)

#reshaping  do it on your own
##reshape ## add a  dimention in the numpy 
#x_train.shape[0] number of rows and x_train.shape[1] num of colomns
x_train =  np.reshape(x_train,(x_train.shape[0],x_train.shape[1], 1)) #batch size total days,  timsteps 60, inputsize #new indicator price of another stock that is dependent


#building the rnna
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout 

#declaring the rnn
regressor = Sequential()


#adding LSTM
regressor.add(LSTM(units =50,return_sequences = True, input_shape =  ( x_train.shape[1], 1 )) )# number of things inside, true if another layers will be added, the last two dimentions of xtrain
#adding regressor.
regressor.add(Dropout(0.2)) #drop of of the neurans 8 will be drop uout of 10. this is the regulization

#second layer
regressor.add(LSTM(units =50,return_sequences = True)) # no input shape only in the first
regressor.add(Dropout(0.2)) #drop of of the neurans 8 will be drop uout of 10. this is the regulization



# third layer
regressor.add(LSTM(units =50,return_sequences = True ) )
regressor.add(Dropout(0.2)) #drop of of the neurans 8 will be drop uout of 10. this is the regulization

#four layer
regressor.add(LSTM(units =50) )
regressor.add(Dropout(0.2)) #drop of of the neurans 8 will be drop uout of 10. this is the regulization

#final layer
regressor.add(Dense(units = 1)) #uints =dimntion of the output

#compile the net
regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

# training it 
regressor.fit(x_train,y_train, epochs = 100, batch_size=32 )

#getting the real values 
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price =  dataset_test.iloc[:,1:2].values


#getting the predicted stock prices of january 2017
#axis =0 for a verilital horiszontal concat
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0) #colomns you want, concat the lines the result to the 
inputs = dataset_total[len(dataset_total) -len(dataset_test) - 60: ].values #all the inputs of january 2017
inputs = inputs.reshape(-1,1) # stock minus january 
inputs = sc.transform(inputs)

X_test = []
for i in range(60,80):   # test has 20 days
    X_test.append(inputs[i-60:i,0]) #zero colomn and 60 rows for each day 
    
X_test = np.array(X_test)
X_test =  np.reshape(X_test,(X_test.shape[0],X_test.shape[1], 1)) #batch size total days,  timsteps 60, inputsize #new indicator price of another stock that is dependent

predicted_stock_price = regressor.predict(X_test)
#go back to non scaling the data
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


#using the matplot to plot the data 
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price' ) # data and label to it 
plt.plot(predicted_stock_price, color = 'red', label = 'Real Google Stock Price' ) #data and label to it
plt.title("google stock price prediction" ) # title 
plt.xlabe('Time')
plt.ylabe('google stock price')
plt.legend() # to includ the legend in the char with no input 
plt.show()

## we can increased its accurary by changing the scoring method  to accuracy or neg_mean_squared_error

