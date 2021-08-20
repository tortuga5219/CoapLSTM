import os
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def createTrainData(xData, step):
    m = np.arange(len(xData) - step)
    x = [stepsizefunc(i, xData, step) for i in m]
    x_batch = np.reshape(np.array(x), (len(m), step, 1))

    y = [stepsizefunc(i, xData, step, 1) for i in m + 1]
    y_batch = np.reshape(np.array(y), (len(m), 1))

    return x_batch, y_batch


def stepsizefunc(i, xData, step, flag=0):
    a = xData[i:(i + step)]
    if flag == 0:
        return a
    else:
        return a[-1]


data = np.sin(2 * np.pi * 0.03 + np.arange(1001)) + np.random.random(1001)
nInput = 1
nOutput = 1
nStep = 20
nHidden = 50

x, y = createTrainData(data, nStep)

xInput = Input(batch_shape=(None, nStep, 1))
xLstm = LSTM(nHidden)(xInput)
xOutput = Dense(nOutput)(xLstm)
model = Model(xInput, xOutput)
model.compile(loss='mse', optimizer=Adam(lr=0.01))

h = model.fit(x, y, epochs=100, batch_size=100, shuffle=True)

nFuture = 20
if len(data) > 100:
    lastData = np.copy(data[-100:])
else:
    lastData = np.copy(data)
dx = np.copy(lastData)
estimate = [dx[-1]]
for i in range(nFuture):
    px = dx[-nStep:].reshape(1, nStep, 1)
    yHat = model.predict(px)[0][0]
    estimate.append(yHat)
    dx = np.insert(dx, len(dx), yHat)

plt.figure(figsize=(8, 4))
plt.plot(h.history['loss'], color='red')
plt.title("Loss History")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

ax1 = np.arange(1, len(lastData) + 1)
ax2 = np.arange(len(lastData), len(lastData) + len(estimate))
plt.figure(figsize=(8, 4))
plt.plot(ax1, lastData, 'b-o', color='blue', markersize=3, label='Time series', linewidth=1)
plt.plot(ax2, estimate, 'b-o', color='red', markersize=3, label='Estimate')
plt.axvline(x=ax1[-1],  linestyle='dashed', linewidth=1)
plt.legend()
plt.show()
