import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np

# import np
import matplotlib.pyplot as plt
# import pandas
import math
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Convert an array of values into a dataset matrix
def dataset_generate(data, step_size=1):
    dataX, dataY = [], []
    for i in range(len(data)- step_size -1):
        a = data[i:(i+ step_size), 0]
        dataX.append(a)
        dataY.append(data[i + step_size, 0])
    return np.array(dataX), np.array(dataY)

# read in data
print("loading data...")
x = []
y = []
with open('data.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        # print("x = " + row[0] + " y = " + row[1])
        # x.append(row[0])
        x.append(pd.to_datetime(row[0]))
        if (row[1] == ""):
            # print("x = " + row[0] + ": NaN")
            y.append(np.nan)
        else:
            y.append(float(row[1]))

df = pd.Series(i for i in y)
y = df.interpolate(method='linear')
# df.fillna('ffill')
# df = pd.Series([1, 2, 3, np.nan, 5])
# y = df.fillna(method='ffill')
# print(*y, sep = ", ")   
# print(*y, sep = ", ")  
y = y.tolist()       
print("total num of points = " + str(len(x)))
xx = []
yy = []
for i in range(0, 365):
    # day = int(math.floor(i / 96))
    xx.append(x[i * 96])
    sum = 0
    for j in range(0, 96):
        # print("i: ", i, " j: ", j, " index: ", i * 96 + j)
        sum += y[i * 96 + j]
    yy.append(sum)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
arr_yy = np.asarray(yy);
# print(*scaled_y, sep = ", ")   
dataset = scaler.fit_transform(arr_yy.reshape(-1, 1))
# dataset[321 + 29] = 0.9
# print(*dataset, sep = ", ")   

# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print("train data size: ", len(train), " test data size: ", len(test))

# Reshape into X=t and Y=t+1
step_size = 30
trainX, trainY = dataset_generate(train, step_size)
testX, testY = dataset_generate(test, step_size)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# # Create and fit the LSTM network
# model = Sequential()
# model.add(LSTM(4, input_dim= step_size))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, nb_epoch=10 
#     0, batch_size=1, verbose=2)

# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loading model from disk")

# show train result
trainPredict = loaded_model.predict(trainX)	
print("trainPredict size: ",trainPredict.shape);
print("trainPredict max: ",np.ndarray.max(trainPredict, axis=0)," trainPredict min: ",np.ndarray.min(trainPredict, axis=0));

# show test result
testPredict = loaded_model.predict(testX)	
print("testPredict size: ",testPredict.shape);
print("testPredict max: ",np.ndarray.max(testPredict, axis=0)," testPredict min: ",np.ndarray.min(testPredict, axis=0));

# show test result

# plot
# start_day = 18
# # end_day = 22 
train_start_point = step_size
train_end_point = train_start_point + len(trainPredict) - 1
test_start_point = train_end_point + step_size + 1
test_end_point = test_start_point + len(testPredict) - 1
# scaled_y = dataset;
# plt.plot(xx,dataset, label='data')
# plt.plot(xx[train_start_point:train_end_point],trainPredict[0:len(trainPredict) - 1], label='train prediction')
# plt.plot(xx[test_start_point:test_end_point],testPredict[0:len(testPredict) - 1], label='test prediction')
test_idx = 9;
test_start = train_end_point + test_idx + step_size + 1;
# print("test_start: ", test_start + 29)
# print("test_start: ", dataset[test_start + 29])
plt.plot(xx[test_start:test_start + 30],dataset[test_start:test_start + 30], 'bo-', label='past data')
plt.plot(xx[test_start + 30],testPredict[0 + test_idx], 'ro', label='prediction')
plt.xlabel('Date')
plt.ylabel('Energy Consumption')
plt.title('Transportation Carbon Emission Prediction')
# plt.title('Energy Consumption' + " from day " + str(start_day) + " to day " + str(end_day))
plt.legend(loc=2)
plt.show()