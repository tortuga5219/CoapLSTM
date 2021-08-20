# !pip install yfinance
import pandas as pd
import yfinance as yf
import math
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def LSTM_model(X_train, y_train, X_test, sc):
    # create a model
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN, GRU, LSTM
    from keras.optimizers import SGD

    # The LSTM architecture
    my_LSTM_model = Sequential()
    my_LSTM_model.add(LSTM(units=50,
                           return_sequences=True,
                           input_shape=(X_train.shape[1], 1),
                           activation='tanh'))
    my_LSTM_model.add(LSTM(units=50, activation='tanh'))
    my_LSTM_model.add(Dense(units=2))

    # Compiling
    my_LSTM_model.compile(optimizer=SGD(lr=0.01, decay=1e-7,
                                        momentum=0.9, nesterov=False),
                          loss='mean_squared_error')

    # Fitting to the training set
    my_LSTM_model.fit(X_train, y_train, epochs=50, batch_size=150, verbose=0)

    my_LSTM_model.summary()

    LSTM_prediction = my_LSTM_model.predict(X_test)
    LSTM_prediction = sc.inverse_transform(LSTM_prediction)

    return my_LSTM_model, LSTM_prediction


def GRU_model(X_train, y_train, X_test, sc):
    # create a model
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN, GRU, Dropout
    from keras.optimizers import SGD

    # The LSTM architecture
    my_GRU_model = Sequential()
    my_GRU_model.add(GRU(units=50,
                          return_sequences=True,
                          input_shape=(X_train.shape[1], 1),
                          activation='tanh'))
    my_GRU_model.add(GRU(units=50, activation='tanh'))
    my_GRU_model.add(Dropout(0.2))
    my_GRU_model.add(Dense(units=2))

    # Compiling
    my_GRU_model.compile(optimizer=SGD(lr=0.01, decay=1e-7,
                                       momentum=0.9, nesterov=False),
                         loss='mean_squared_error')

    # Fitting to the training set
    my_GRU_model.fit(X_train, y_train, epochs=50, batch_size=150, verbose=0)

    my_GRU_model.summary()

    GRU_prediction = my_GRU_model.predict(X_test)
    GRU_prediction = sc.inverse_transform(GRU_prediction)

    return my_GRU_model, GRU_prediction

def actual_pred_plot(preds):
    """
    Plot the actual vs predition
    """
    actual_pred = pd.DataFrame(columns=['Adj. Close', 'prediction'])
    actual_pred['Adj. Close'] = all_data.loc['2019':, 'Adj Close'][0:len(preds)]
    actual_pred['prediction'] = preds[:, 0]

    from keras.metrics import MeanSquaredError
    m = MeanSquaredError()
    m.update_state(np.array(actual_pred['Adj. Close']), np.array(actual_pred['prediction']))

    return (m.result().numpy(), actual_pred.plot())


def ts_train_test_normalize(all_data, time_steps, for_periods):
    """
    input:
        data: dataframe with dates and price data
    output:
        X_train, y_train: data from 2013/1/1-2018/12/31
        X_test : data from 2019-
        sc :     insantiated MinMaxScaler object fit to the training data
    """
    # create training and test set
    ts_train = all_data[:'2018'].iloc[:, 0:1].values
    ts_test = all_data['2019':].iloc[:, 0:1].values
    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)

    # scale the data
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0, 1))
    ts_train_scaled = sc.fit_transform(ts_train)

    # create training data of s samples and t time steps
    X_train = []
    y_train = []
    for i in range(time_steps, ts_train_len - 1):
        X_train.append(ts_train_scaled[i - time_steps:i, 0])
        y_train.append(ts_train_scaled[i:i + for_periods, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping X_train for efficient modelling
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    inputs = pd.concat((all_data["Adj Close"][:'2018'], all_data["Adj Close"]['2019':]), axis=0).values
    inputs = inputs[len(inputs) - len(ts_test) - time_steps:]
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    # Preparing X_test
    X_test = []
    for i in range(time_steps, ts_test_len + time_steps - for_periods):
        X_test.append(inputs[i - time_steps:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, sc


AMZN = yf.download('AMZN',
                   start='2013-01-01',
                   end='2019-12-31',
                   progress=False)
all_data = AMZN[['Adj Close', 'Open', 'High', 'Low', "Close", "Volume"]].round(2)

X_train, y_train, X_test, sc = ts_train_test_normalize(all_data, 5, 2)
# my_rnn_model, rnn_predictions_2 = simple_rnn_model(X_train, y_train, X_test, sc)
# rnn_res2 = rnn_predictions_2[1:10]
# actual_pred_plot(rnn_res2)



# my_LSTM_model, LSTM_prediction = LSTM_model(X_train, y_train, X_test, sc)
# LSTM_prediction[1:10]
# actual_pred_plot(LSTM_prediction)
# plt.title("LSTM")

my_GRU_model, GRU_prediction = GRU_model(X_train, y_train, X_test, sc)
GRU_prediction[1:10]
actual_pred_plot(GRU_prediction)
plt.title("GRU")


plt.show()
