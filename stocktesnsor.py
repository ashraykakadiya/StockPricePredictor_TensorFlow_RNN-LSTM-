import numpy as npy
from keras.models import Sequential
import pandas as pnd
from pandas import datetime
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import math, time
import itertools
from math import sqrt
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plot



def get_stock_data(stock_name, normalized=0):
    url="http://finance.google.com/finance/historical?cid=22144&startdate=Jan+1%2C+2011&enddate=Dec+2%2C+2017&num=30&ei=l8IiWtD8L8SnjAH-hbDABw&output=csv"


    col_names = ['Date','Open','High','Low','Close','Volume']
    stocks = pnd.read_csv(url, header=0, names=col_names)
    data = pnd.DataFrame(stocks)
    data.drop(data.columns[[0,3,5]], axis=1, inplace=True)
    return data

stock_name = 'AAPL'
data = get_stock_data(stock_name,0)
data.tail()

today = datetime.date.today()
f_name = stock_name+'_stock_%s.csv' % today
data.to_csv(f_name)

data['High'] = data['High'] / 1000
data['Open'] = data['Open'] / 1000
data['Close'] = data['Close'] / 1000
data.head(5)

def loading_data(stocks, seq_len):
    amount_of_features = len(stocks.columns)
    all_stocks = stocks.as_matrix() #pnd.DataFrame(stocks)
    sq_len = seq_len + 1
    final_result = []
    for index in range(len(all_stocks) - sq_len):
        final_result.append(all_stocks[index: index + sq_len])

    final_result = npy.array(final_result)
    row = round(0.9 * final_result.shape[0])
    train = final_result[:int(row), :]
    train_X = train[:, :-1]
    train_Y = train[:, -1][:,-1]
    test_X = final_result[int(row):, :-1]
    test_Y = final_result[int(row):, -1][:,-1]

    train_X = npy.reshape(train_X, (train_X.shape[0], train_X.shape[1], amount_of_features))
    test_X = npy.reshape(test_X, (test_X.shape[0], test_X.shape[1], amount_of_features))

    return [train_X, train_Y, test_X, test_Y]

def modelbuild(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[2]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

def modelbuild2(layers):
        d = 0.2
        model = Sequential()
        model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(Dropout(d))
        model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
        model.add(Dropout(d))
        model.add(Dense(16,init='uniform',activation='relu'))
        model.add(Dense(1,init='uniform',activation='relu'))
        model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
        return model

win = 5
train_X, train_Y, test_X, test_Y = loading_data(data[::-1], win)
print("train_X", train_X.shape)
print("train_Y", train_Y.shape)
print("test_X", test_X.shape)
print("test_Y", test_Y.shape)

# model = modelbuild([3,lag,1])
model = modelbuild2([3,win,1])
model.fit(
    train_X,
    train_Y,
    batch_size=512,
    nb_epoch=500,
    validation_split=0.1,
    verbose=0)

trainScore = model.evaluate(train_X, train_Y, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(test_X, test_Y, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

# print(test_X[-1])
diff=[]
ratio=[]
p = model.predict(test_X)
for u in range(len(test_Y)):
    pr = p[u][0]
    ratio.append((test_Y[u]/pr)-1)
    diff.append(abs(test_Y[u]- pr))
    #print(u, test_Y[u], pr, (test_Y[u]/pr)-1, abs(test_Y[u]- pr))

import matplotlib.pyplot as plot2

plot2.plot(p,color='red', label='prediction')
plot2.plot(test_Y,color='blue', label='test_Y')
plot2.legend(loc='upper left')
plot2.show()