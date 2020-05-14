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

# Binary Classification 계산을 위해 t값을 각각의 class 에서 0,1 으로 바꾼다.
t_bin_train = list()
for i in range(label_name.shape[0]):
    t_bin_train.append([j == i for j in t_train])
t_bin_train = np.array(t_bin_train)

t_bin_test = list()
for i in range(label_name.shape[0]):
    t_bin_test.append([j == i for j in t_test])
t_bin_test = np.array(t_bin_test)

for i in range(label_name.shape[0]):
    LR_mnist = LR.LogisticRegressionBinary(x_train, t_bin_train[i], label_name)
    LR_mnist.learn(100, 0.001)
    LR_mnist.predict(x_test, t_bin_test[i])
