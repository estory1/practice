#%% [markdown]
# # Stock Price Prediction
# ### Date created: 20190505
#
# Borrows from: https://github.com/Kulbear/stock-prediction/blob/master/stock-prediction.ipynb
#
#%%
import os
import time
import math
import pandas as pd
import numpy as np

import sklearn.preprocessing as prep

%matplotlib inline 

import matplotlib
import matplotlib.pyplot as plt

#%% [markdown]
# ## Data Eng
#%%
df = pd.read_csv(
    os.path.expanduser('~') + "/home/financial/finance.yahoo.com_data/20190514 - finance.yahoo.com - GSPC -SP500 index - daily - 1950-20190514.csv",
    index_col=0,
    parse_dates=[0])

# reorder cols so target is in last position
df = df[ [ "Open", "High", "Low", "Close", "Volume", "Adj Close"] ]
df["d(Adj Close)"] = df["Adj Close"].diff()
df["d(Adj Close)"].fillna(0, inplace=True)

df.iloc[ list(range(0,3)) + list(range(-4,-1)) ]

#%%
df.dtypes

#%%
df.describe()

#%%
# standardize without scaling train & test together (avoid information leak).
def standard_scaler(X_train, X_test):
    train_samples, train_nx, train_ny = X_train.shape
    test_samples, test_nx, test_ny = X_test.shape
    
    X_train = X_train.reshape((train_samples, train_nx * train_ny))
    X_test = X_test.reshape((test_samples, test_nx * test_ny))
    
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    X_train = X_train.reshape((train_samples, train_nx, train_ny))
    X_test = X_test.reshape((test_samples, test_nx, test_ny))
    
    return X_train, X_test

# Not a great preprocessor, but for a first RNN it'll do.
def preprocess_data(stock, seq_len):
    feature_count = len(stock.columns)
    data = stock.values
    
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index : index + sequence_length])
        
    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[: int(row), :]
    
    train, result = standard_scaler(train, result)
    
    X_train = train[:, : -1]
    y_train = train[:, -1][: ,-1]
    X_test = result[int(row) :, : -1]
    y_test = result[int(row) :, -1][ : ,-1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], feature_count))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], feature_count))  

    return [X_train, y_train, X_test, y_test]

window = 30
# X_train, y_train, X_test, y_test = preprocess_data(df[:: -1], window)
X_train, y_train, X_test, y_test = preprocess_data(df, window)  # original author's code passed df[:: -1] because their dataset was sorted time-descending; mine is time-ascending.
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)


#%% [markdown]
# ## Model
#
# 20240219 note: This section is un-runnable because I have Python 3.11, but Keras depends on Tensorflow, which currently requires <= Python 3.10.

#%%
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

#%%
def build_model(layers):
    model = Sequential()

    # By setting return_sequences to True we are able to stack another LSTM layer
    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    # model.add(Dropout(0.4))
    model.add(Dropout(0.2))
    # model.add(Dropout(0.1))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    # model.add(Dropout(0.3))
    model.add(Dropout(0.15))
    # model.add(Dropout(0.08))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))
    # model.add(Activation("elu"))    # https://keras.io/layers/advanced-activations/
    # model.add(Activation("exponential"))  # https://keras.io/activations/

    start = time.time()
    # model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    # model.compile(loss="mse", optimizer="adam", metrics=['mse'])  # accuracy is a classifier metric, but we're predicting a continuous variable...
    model.compile(
        loss="mse",
        optimizer="adam",
        metrics=['mse', 'mae', 'kullback_leibler_divergence'])
    print("Compilation Time : ", time.time() - start)
    return model

#%%
# Build the model.
# model = build_model([X_train.shape[2], window, 100, 1])
model = build_model([X_train.shape[2], window, 100*5, 1])

# Train the model.
model.fit(
    X_train,
    y_train,
    batch_size=768,
    epochs=10,
    validation_split=0.1,
    verbose=1)

#%%
trainScore = model.evaluate(X_train, y_train, verbose=1)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=1)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

#%% [markdown]
# ## Visualize
#%%
diff = []
ratio = []
pred = model.predict(X_test)
for u in range(len(y_test)):
    pr = pred[u][0]
    ratio.append((y_test[u] / pr) - 1)
    diff.append(abs(y_test[u] - pr))

#%%
plt.plot(pred, color='red', label='Prediction')
plt.plot(y_test, color='blue', label='Ground Truth')
plt.plot(ratio, color='green', label='Ratio: pred/actual - 1')
plt.legend(loc='upper left')
plt.show()






#%% [markdown]
# ## Predict Adj Price direction instead of value.
#%%
df2 = df.copy()
df2["direction"] = df2["d(Adj Close)"] >= 0  # True = 0 or up; False = down
df2.iloc[ list(range(0,3)) + list(range(-4,-1)) ]

#%%
window = 30
# X_train, y_train, X_test, y_test = preprocess_data(df[:: -1], window)
X_train, y_train, X_test, y_test = preprocess_data(df2, window)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)

#%%
def build_model(layers):
    model = Sequential()

    # By setting return_sequences to True we are able to stack another LSTM layer
    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    # model.add(Dropout(0.4))
    model.add(Dropout(0.2))
    # model.add(Dropout(0.1))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    # model.add(Dropout(0.3))
    model.add(Dropout(0.15))
    # model.add(Dropout(0.08))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("relu"))
    # model.add(Activation("elu"))    # https://keras.io/layers/advanced-activations/
    # model.add(Activation("exponential"))  # https://keras.io/activations/

    start = time.time()
    # model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    # model.compile(loss="mse", optimizer="adam", metrics=['mse'])  # accuracy is a classifier metric, but we're predicting a continuous variable...
    model.compile(
        loss="mse",
        optimizer="adam",
        metrics=['mse', 'accuracy', 'binary_accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

#%%
# Build the model.
# model = build_model([X_train.shape[2], window, 100, 1])
model = build_model([X_train.shape[2], window, 100*5, 1])

# Train the model.
model.fit(
    X_train,
    y_train,
    batch_size=768,
    epochs=10,
    validation_split=0.1,
    verbose=1)

#%%
trainScore = model.evaluate(X_train, y_train, verbose=1)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=1)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

#%% [markdown]
# ## Visualize
#%%
diff = []
ratio = []
pred = model.predict(X_test)
for u in range(len(y_test)):
    pr = pred[u][0]
    ratio.append((y_test[u] / pr) - 1)
    diff.append(abs(y_test[u] - pr))

#%%
plt.plot(pred, color='red', label='Prediction')
plt.plot(y_test, color='blue', label='Ground Truth')
plt.plot(ratio, color='green', label='Ratio: pred/actual - 1')
plt.legend(loc='upper left')
plt.show()


#%%
