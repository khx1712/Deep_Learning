import sys, os
sys.path.append(os.pardir)  # 현재 폴더의 부모 폴더에서 찾아서 loading
import numpy as np
from sklearn.datasets import load_iris
from LR import Logistic_Regression as LR

iris = load_iris()  # scikit-learn 에서 iris data 를 불러와 저장

X = iris.data  # 150*4 의 iris data 를 가져온다
bias = np.ones((X.shape[0], 1))  # bias 를 X에 추가하기 위해 (150,1) array 를 1로 초기화 시켜서 만든다.
X = np.hstack((X, bias))  # bias 를 X의 가장 끝 열에 추가한다 (150,4) -> (150,5)
y = iris.target  # 150*1 의 iris 의 실제 값을 가져온다
y_name = iris.target_names  # target 이름 저장

#  Multinomial Classification 계산을 위해 y값을 one-hot encoding
num = np.unique(y, axis=0)
num = num.shape[0]
y_oneHot = np.eye(num)[y]  # (150,1) array 를 (150,3) array 로 one-hot encoding 한다.

cycle = 4  # 얼만큼의 주기로 test data 를 만들껀지 정한다.
for_test = np.array([(i % cycle == (cycle - 1)) for i in range(y.shape[0])])
# y.shape[0] 개수 150 / 주기에 해당하는 지점만 1으로 초기화해서 test array 를 만든다
for_train = ~for_test  # test 에 해당하는 지점을 제외한 점들을 모두 1로 만든다.

X_train = X[for_train]  # for_train 에 해당하는 지점의 값들을 잘라서 list 로 만든다
y_mul_train = y_oneHot[for_train]  # for_train 에 해당하는 지점의 값들을 잘라서 list 로 만든다

X_test = X[for_test]  # for_test 에 해당하는 지점의 값들을 잘라서 list 로 만든다
y_mul_test = y_oneHot[for_test]  # for_test 에 해당하는 지점의 값들을 잘라서 list 로 만든다

LR_iris_mul = LR.LogisticRegression(X_train, y_mul_train, y_name)  # X, y, target 값을 argument 로 넘겨 초기화 시킨 class 를 생성한다.
LR_iris_mul.learn(500, 0.001)  # epoch, learning rate 값을 argument 로 넘겨 class 내부에 있는 train data 로 학습시킽다.
LR_iris_mul.predict(X_test, y_mul_test)  # X test, y test 값을 argument 로 넘겨 몇 퍼센트 적중률을 가지는지 test 한다.
