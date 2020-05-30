import sys, os
sys.path.append(os.pardir)  # 현재 폴더의 부모 폴더에서 찾아서 loading
import numpy as np
from dataset.mnist import load_mnist
from LR import Logistic_Regression as LR

# mnist data 를 load 하는 함수
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, flatten=True)

bias = np.ones((x_train.shape[0], 1))  # bias 를 X에 추가하기 위해 (60000,1) array 를 1로 초기화 시켜서 만든다.
x_train = np.hstack((x_train, bias))  # bias 를 X의 가장 끝 열에 추가한다 (60000,784) -> (60000,785)
bias = np.ones((x_test.shape[0], 1))  # bias 를 X에 추가하기 위해 (10000,1) array 를 1로 초기화 시켜서 만든다.
x_test = np.hstack((x_test, bias))  # bias 를 X의 가장 끝 열에 추가한다 (10000,784) -> (10000,785)

label_name = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])  # target 을 정의해준다.

# Binary Classification 계산을 위해 t값을 각각의 class 에서 0,1 으로 바꾼다.
t_bin_train = list()
for i in range(label_name.shape[0]):
    t_bin_train.append([j == i for j in t_train])
t_bin_train = np.array(t_bin_train)

t_bin_test = list()
for i in range(label_name.shape[0]):
    t_bin_test.append([j == i for j in t_test])
t_bin_test = np.array(t_bin_test)

for i in range(label_name.shape[0]):  # 0~9의 target 각각에 Binary Classification 을 수행한다.
    # X, t, target 값을 argument 로 넘겨 초기화 시킨 Multinomial class 를 생성한다.
    LR_mnist = LR.LogisticRegressionBinary(x_train, t_bin_train[i], label_name)
    LR_mnist.learn(100, 0.0001)  # epoch, learning rate 값을 argument 로 넘겨 class 내부에 있는 train data 로 학습시킽다.
    LR_mnist.predict(x_test, t_bin_test[i])  # X test, t test 값을 argument 로 넘겨 몇 퍼센트 적중률을 가지는지 test 한다.

