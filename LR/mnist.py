import sys, os
sys.path.append(os.pardir)  # 현재 폴더의 부모 폴더에서 찾아서 loading
import numpy as np
from dataset.mnist import load_mnist
from LR import Logistic_Regression as LR

# mnist data를 load 하는 함수
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, flatten=True)

bias = np.ones((x_train.shape[0], 1))
x_train = np.hstack((x_train, bias))
bias = np.ones((x_test.shape[0], 1))
x_test = np.hstack((x_test, bias))

label_name = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

#  Multinomial Classification 계산을 위해 y값을 one-hot encoding
num = np.unique(t_train, axis=0)
num = num.shape[0]
t_oneHot_train = np.eye(num)[t_train]  # 150*3

num = np.unique(t_test, axis=0)
num = num.shape[0]
t_oneHot_test = np.eye(num)[t_test]  # 150*3

LR_mnist_mul = LR.LogisticRegression(x_train, t_oneHot_train, label_name)
LR_mnist_mul.learn(500, 0.001)
LR_mnist_mul.predict(x_test, t_oneHot_test)
