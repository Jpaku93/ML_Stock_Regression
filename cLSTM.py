from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd   

# split data into train and test and validation
def split_data_LSTM(data,train_size = 0.7, val_size = 0.2):
    train = data[:int(len(data)* train_size)]
    val = data[int(len(data)*train_size):int(len(data)*(train_size+val_size))]
    test = data[int(len(data)*(train_size+val_size)):]
    return train, val, test


def scale_data(train, val, test):
    scaler = MinMaxScaler()
    scaler.fit(train.values.reshape(-1, 1))
    train_scaled = scaler.transform(train.values.reshape(-1, 1))
    test_scaled = scaler.transform(test.values.reshape(-1, 1))
    val_scaled = scaler.transform(val.values.reshape(-1, 1))
    return train_scaled, val_scaled, test_scaled, scaler

def create_sequences(data, WINDOW_SIZE):
    xs = []
    ys = []
    for i in range(len(data)-WINDOW_SIZE-1):
        x = data[i:(i+WINDOW_SIZE)]
        y = data[i+WINDOW_SIZE]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
    
def prepare_data(data, WINDOW_SIZE = 5):
    train, val, test = split_data_LSTM(data)
    train_scaled, val_scaled, test_scaled, scaler = scale_data(train, val, test)
    X_train_scaled, y_train_scaled = create_sequences(train_scaled,WINDOW_SIZE)
    X_test_scaled, y_test_scaled = create_sequences(test_scaled,WINDOW_SIZE)
    X_val_scaled, y_val_scaled = create_sequences(val_scaled,WINDOW_SIZE)    
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, X_val_scaled, y_val_scaled, scaler

def build_LSTM(WINDOW_SIZE = 5):
    model = Sequential()
    model.add(LSTM(units=6, return_sequences=True, input_shape=(WINDOW_SIZE, 1)))
    model.add(LSTM(units=12, return_sequences=True))
    model.add(LSTM(units=24, return_sequences=True))
    model.add(Dense(1))
    model.summary()
    return model

    