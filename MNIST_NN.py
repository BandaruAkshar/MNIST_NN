import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('mnist_train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

dataDev = data[0:1000].T

Y_test = dataDev[0]
X_test = dataDev[1:n]
X_test = X_test / 255

dataTrain = data[1000:m].T

Y_train = dataTrain[0]
X_train = dataTrain[1:n]
X_train = X_train / 255


def initParams():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2


def ReLU(z):
    return np.maximum(0, z)


def softmax(z):
    val1 = np.exp(z) / sum(np.exp(z))
    return val1


def forwardProp(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)  # ReLU activation fn
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)  # softmax activation fn
    return z1, a1, z2, a2


def ReLU1(z):
    return z > 0


def oneHotEncode(y):
    oneHotY = np.zeros((y.size, y.max() + 1))
    oneHotY[np.arange(y.size), y] = 1
    oneHotY = oneHotY.T
    return oneHotY


def backProp(z1, a1, z2, a2, w1, w2, x, y):
    oneHotY = oneHotEncode(y)
    dz2 = a2 - oneHotY
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * ReLU1(z1)
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1)
    return dw1, db1, dw2, db2


def updateParameters(w1, b1, w2, b2, dw1, db1, dw2, db2, a):
    w1 = w1 - a * dw1
    b1 = b1 - a * db1
    w2 = w2 - a * dw2
    b2 = b2 - a * db2
    return w1, b1, w2, b2


def getPredVal(a2):
    return np.argmax(a2, 0)


def getAccVal(predictions, y):
    return np.sum(predictions == y) / y.size


def gradientDescent(x, y, a, iters):
    w1, b1, w2, b2 = initParams()

    for i in range(iters):
        z1, a1, z2, a2 = forwardProp(w1, b1, w2, b2, x)
        dW1, db1, dW2, db2 = backProp(z1, a1, z2, a2, w1, w2, x, y)
        w1, b1, w2, b2 = updateParameters(w1, b1, w2, b2, dW1, db1, dW2, db2, a)

        if i == iters-1:
            predictions = getPredVal(a2)
            print(f'Training Accuracy:{str(getAccVal(predictions, y))}')
    return w1, b1, w2, b2


def makePred(x, w1, b1, w2, b2):
    _, _, _, a2 = forwardProp(w1, b1, w2, b2, x)  # forwardProp returns z1, a1, z2, a2
    pred = getPredVal(a2)
    return pred


def testPredictions(index, w1, b1, w2, b2):
    currentImage = X_train[:, index, None]
    pred = makePred(X_train[:, index, None], w1, b1, w2, b2)
    label = Y_train[index]
    print("Prediction: ", pred)
    print("Label: ", label)

    currentImage = currentImage.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(currentImage, interpolation='nearest')
    plt.show()


w1, b1, w2, b2 = gradientDescent(X_train, Y_train, 0.10, 500)
devPredictions = makePred(X_test, w1, b1, w2, b2)
print(f'Test Accuracy:{str(getAccVal(devPredictions, Y_test))}')
