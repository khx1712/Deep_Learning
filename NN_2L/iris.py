import numpy as np
from sklearn.datasets import load_iris
import sys, os
sys.path.append(os.pardir)  # 현재 폴더의 부모 폴더에서 찾아서 loading
from NN_2L import Neural_Network as NN

iris = load_iris()  # scikit-learn 에서 iris data 를 불러와 저장

X = iris.data  # 150*4
y = iris.target  # 150*1
y_name = iris.target_names  # target 이름 저장

#  Multinomial Classification 계산을 위해 y값을 one-hot encoding
num = np.unique(y, axis=0)
num = num.shape[0]
y_oneHot = np.eye(num)[y]  # (150,1) array 를 (150,3) array 로 one-hot encoding 한다.

cycle = 5  # 얼만큼의 주기로 test data 를 만들껀지 정한다.
for_test = np.array([(i % cycle == (cycle - 1)) for i in range(y.shape[0])])
# y.shape[0] 개수 150 / 주기에 해당하는 지점만 1으로 초기화해서 test array 를 만든다
for_train = ~for_test  # test 에 해당하는 지점을 제외한 점들을 모두 1로 만든다.

X_train = X[for_train]  # train 에 해당하는 지점의 값들으로 list 만든다
y_train = y_oneHot[for_train]  # train 에 해당하는 지점의 값들으로 list 만든다

X_test = X[for_test]  # test 에 해당하는 지점의 값들으로 list 만든다
y_test = y_oneHot[for_test]  # test 에 해당하는 지점의 값들으로 list 만든다

input_size = 4  # feature 의 개수 만큼 input layer 의 unit 의 개수를 정한다.
hidden_size = 4 # hidden layer 에 몇개의 unit을 만들어줄껀지 정해준다.
output_size = 3  # 마지막에 softmax 를 통해 확률값을 구해야 하므로 target 의 개수로 정의한다.
NeuralNet_iris = NN.TwoLayerNet(input_size, hidden_size, output_size)  # 위에서 정한 각 layer 의 unit 의 개수를 입력하여 class 를 생성한다.
NeuralNet_iris.learn(0.015, 15000, 10)  # learning rate/epoch/batch size 를 입력하여 학습을 진행한다.
print('Test Accuracy = ', NeuralNet_iris.accuracy(X_test, y_test))  # 학습을 모두 마치고 test data 를 입력하여 accuracy 를 통해 얼마나 학습이 잘됬는지 확인한다.

