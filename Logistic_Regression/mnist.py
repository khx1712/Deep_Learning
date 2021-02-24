import sys, os
sys.path.append(os.pardir)  # 현재 폴더의 부모 폴더에서 찾아서 loading
import numpy as np
from dataset.mnist import load_mnist
from LR import Logistic_Regression as LR

# mnist data를 load 하는 함수
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, flatten=True)

bias = np.ones((x_train.shape[0], 1))  # bias 를 X에 추가하기 위해 (60000,1) array 를 1로 초기화 시켜서 만든다.
x_train = np.hstack((x_train, bias))  # bias 를 X의 가장 끝 열에 추가한다 (60000,784) -> (60000,785)
bias = np.ones((x_test.shape[0], 1))  # bias 를 X에 추가하기 위해 (10000,1) array 를 1로 초기화 시켜서 만든다.
x_test = np.hstack((x_test, bias))  # bias 를 X의 가장 끝 열에 추가한다 (10000,784) -> (10000,785)

label_name = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])  # target 을 정의해준다.

#  Multinomial Classification 계산을 위해 y값을 one-hot encoding
num = np.unique(t_train, axis=0)
num = num.shape[0]
t_oneHot_train = np.eye(num)[t_train]  # (60000,1) array 를 (60000,10) array 로 one-hot encoding 한다.
num = np.unique(t_test, axis=0)
num = num.shape[0]
t_oneHot_test = np.eye(num)[t_test]  # (10000,1) array 를 (10000,10) array 로 one-hot encoding 한다.

# 0~9의 target 을 한번에 Multinomial Classification 을 수행한다.
# X, t, target 값을 argument 로 넘겨 초기화 시킨 class 를 생성한다.
LR_mnist_mul = LR.LogisticRegression(x_train, t_oneHot_train, label_name)
LR_mnist_mul.learn(300, 0.0001)  # epoch, learning rate 값을 argument 로 넘겨 class 내부에 있는 train data 로 학습시킽다.
LR_mnist_mul.predict(x_test, t_oneHot_test)  # X test, t test 값을 argument 로 넘겨 몇 퍼센트 적중률을 가지는지 test 한다.

