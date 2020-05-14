import sys, os
sys.path.append(os.pardir)  # 현재 폴더의 부모 폴더에서 찾아서 loading
import numpy as np
from sklearn.datasets import load_iris
from LR import Logistic_Regression as LR

iris = load_iris()  # scikit-learn 에서 iris data 를 불러와 저장

X = iris.data  # 150*4 의 iris data 를 가져온다
bias = np.ones((X.shape[0], 1))  # bias 를 X에 추가하기 위해 (150,1) array 를 1로 초기화 시켜서 만든다.
X = np.hstack((X, bias))  # bias 를 X의 가장 끝 열에 추가한다 (150,4) -> (150,5)
y = iris.target  # 150*1
y_name = iris.target_names  # target 이름 저장

# Binary Classification 계산을 위해 y값을 각각의 class 에서 0,1 으로 바꾼다.
y_bin = list()
for i in range(y_name.shape[0]):
    y_bin.append([j == i for j in y])
y_bin = np.array(y_bin)  # (150,1) array 를 (150,3) array 로 one-hot encoding 한다.

cycle = 4  # 얼만큼의 주기로 test data 를 만들껀지 정한다.
for_test = np.array([(i % cycle == (cycle - 1)) for i in range(y.shape[0])])
# y.shape[0] 개수 150 / 주기에 해당하는 지점만 1으로 초기화해서 test array 를 만든다
for_train = ~for_test  # test 에 해당하는 지점을 제외한 점들을 모두 1로 만든다.

X_train = X[for_train]  # for_train 에 해당하는 지점의 값들을 잘라서 list 로 만든다
y0_train = y_bin[0][for_train]  # for_train 에 해당하는 지점의 값들을 잘라서 list 로 만든다
y1_train = y_bin[1][for_train]  # for_train 에 해당하는 지점의 값들을 잘라서 list 로 만든다
y2_train = y_bin[2][for_train]  # for_train 에 해당하는 지점의 값들을 잘라서 list 로 만든다

X_test = X[for_test]  # for_test 에 해당하는 지점의 값들을 잘라서 list 로 만든다
y0_test = y_bin[0][for_test]  # for_test 에 해당하는 지점의 값들을 잘라서 list 로 만든다
y1_test = y_bin[1][for_test]  # for_test 에 해당하는 지점의 값들을 잘라서 list 로 만든다
y2_test = y_bin[2][for_test]  # for_test 에 해당하는 지점의 값들을 잘라서 list 로 만든다

# target class 0
LR_iris = LR.LogisticRegressionBinary(X_train, y0_train, y_name)  # X, y, target 값을 argument 로 넘겨 초기화 시킨 class 를 생성한다.
LR_iris.learn(1000, 0.001)  # epoch, learning rate 값을 argument 로 넘겨 class 내부에 있는 train data 로 학습시킽다.
LR_iris.predict(X_test, y0_test)  # X test, y test 값을 argument 로 넘겨 몇 퍼센트 적중률을 가지는지 test 한다.

# target class 1
LR_iris = LR.LogisticRegressionBinary(X_train, y1_train, y_name)  # X, y, target 값을 argument 로 넘겨 초기화 시킨 class 를 생성한다.
LR_iris.learn(1000, 0.0001)  # epoch, learning rate 값을 argument 로 넘겨 class 내부에 있는 train data 로 학습시킽다.
LR_iris.predict(X_test, y1_test)  # X test, y test 값을 argument 로 넘겨 몇 퍼센트 적중률을 가지는지 test 한다.

# target class 2
LR_iris = LR.LogisticRegressionBinary(X_train, y2_train, y_name)  # X, y, target 값을 argument 로 넘겨 초기화 시킨 class 를 생성한다.
LR_iris.learn(300, 0.001)  # epoch, learning rate 값을 argument 로 넘겨 class 내부에 있는 train data 로 학습시킽다.
LR_iris.predict(X_test, y2_test)  # X test, y test 값을 argument 로 넘겨 몇 퍼센트 적중률을 가지는지 test 한다.
