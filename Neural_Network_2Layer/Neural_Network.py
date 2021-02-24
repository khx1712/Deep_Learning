import numpy as np
import sys, os
sys.path.append(os.pardir)
from function import sigmoid, softmax, cross_entropy_error_onehot # 기존에 만들어 놨던 함수들을 가져온다.
import matplotlib.pyplot as plt  # cost/accuracy 그래프를 그리기 위해 가져온다.
from sklearn.datasets import load_iris  # iris 에 대한 data 를 가져온다.


class TwoLayerNet:  # 1개의 hidden layer 를 가지고 있는 Neural Network class
    x = np.array([])  # input data
    t = np.array([])  # target data
    params = {}  # weight 와 bias 를 가지는 parameter

    def __init__(self, input_size, hidden_size, output_size):  # 각 layer 의 unit 개수를 받아와 network 를 초기화한다.
        self.input_size = input_size  # input size 저장
        self.output_size = output_size  # output size 저장
        # W1: x(input layer) -> hidden layer
        self.params['W1'] = np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.random.randn(hidden_size)
        # W2: hidden layer -> y(output layer)
        self.params['W2'] = np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.random.randn(output_size)

    def predict(self, x):  # x 가들어가면 현재 가지고 있는 weight 와 bias 값을 가지고 y(output)을 예측한다.
        # x(input layer) -> hidden layer
        z2 = np.dot(x, self.params['W1']) + self.params['b1']
        a2 = sigmoid(z2)  # 계산 값에 sigmoid 함수를 적용한다.
        # hidden layer -> y(output layer)
        z3 = np.dot(a2, self.params['W2']) + self.params['b2']
        y = softmax(z3)  # y값에 softmax 함수를 적용하여 확룰값으로 바꾼다.

        return y  # (10,3)

    def loss(self, x, t):  # 예측 값과 t(target) 값과의 loss 를 계산, 모든 data 에 대한 x, t 한번에 들어옴
        y = self.predict(x)  # (m * sl)
        return cross_entropy_error_onehot(y, t)

    def accuracy(self, x, t):  # 계산된 output의 최대값과 target을 비교해서 일치률을 계산
        y = self.predict(x)
        cnt = 0
        for i in range(y.shape[0]):
            max_idx = np.argmax(y[i])
            if t[i][max_idx] == 1:
                cnt += 1
        return cnt / y.shape[0]  # 몇퍼센트 만큼 적중에 성공했는지 출력

    def numerical_gradient(self, x, t):
        h = 1e-7
        grads = {}  # 편미분 값을 저장하는 딕셔너리 생성
        for key in self.params:  # 모든 딕셔너리의 key 값에 대해서
            grad = np.zeros_like(self.params[key])  # weight 혹은 bias 와 같은 크기의 array 를 만들어준다

            for idx in np.ndindex(self.params[key].shape):  # 해당 딕셔너리의 모든 원소에 편미분을 적용한다.
                tmp_val = self.params[key][idx]  # 원래의 W, b 값을 저장

                self.params[key][idx] = tmp_val + h  # W, b 값에 아주작은 h를 더한다
                fxh1 = self.loss(x, t)  # f(W+h) => W, b 값을 다르게하여 cost 값을 계산한다
                self.params[key][idx] = tmp_val - h  # W, b 값에 아주작은 h를 뺀다
                fxh2 = self.loss(x, t)  # f(W-h) => W, b 값을 다르게하여 cost 값을 계산한다

                grad[idx] = (fxh1 - fxh2) / (2 * h)  # 평균 변화율을 구한다.
                self.params[key][idx] = tmp_val  # 다시 원래의 값으로 바꿔준다

            grads[key] = grad  # 편미분 값을 딕셔너리에 저장
        return grads  # 모든 parameter 에 대한 편미분값을 return

     # Train : Test = 8 : 2 / lr,epoch 바꿔가며 / Batch 크기 알아서 바꿔가며
    def learn(self, lr, epoch, batch_size):  # 실제로 학습을 진행하는 함수
        iris = load_iris()  # scikit-learn 에서 iris data 를 불러와 저장

        X = iris.data  # 150*4
        y = iris.target  # 150*1

        #  Multinomial Classification 계산을 위해 y값을 one-hot encoding
        num = np.unique(y, axis=0)
        num = num.shape[0]
        y_oneHot = np.eye(num)[y]  # (150,1) array 를 (150,3) array 로 one-hot encoding 한다.

        cycle = 5  # 얼만큼의 주기로 test data 를 만들껀지 정한다.
        for_test = np.array([(i % cycle == (cycle - 1)) for i in range(y.shape[0])])
        # y.shape[0] 개수 150 / 주기에 해당하는 지점만 1으로 초기화해서 test array 를 만든다
        for_train = ~for_test  # test 에 해당하는 지점을 제외한 점들을 모두 1로 만든다.

        self.x = X[for_train]  # train 에 해당하는 지점의 값들으로 list 만든다
        self.t = y_oneHot[for_train]  # train 에 해당하는 지점의 값들으로 list 만든다

        batch_size = max(batch_size, self.x.shape[0])  # 몇개의 data 를 동시에 학습시키는지 size 를 정해준다
        cost_list = list()  # 그래프로 나타내기 위해 cost 값을 저장 하는 list 를 생성
        accuracy_list = list()  # 그래프로 나타내기 위해 accuracy 값을 저장 하는 list 를 생성
        for epo in range(epoch):  # epoch 만큼 학습은 반복한다.
            batch_mask = np.random.choice(self.x.shape[0], batch_size)  # 0 ~ m(data 개수) 사이의 수를 batch_size 개 뽑는다.
            x_batch = self.x[batch_mask]  # (batch_size, 4)
            t_batch = self.t[batch_mask]  # (batch_size, 3)
            cost = self.loss(self.x, self.t)  # test data 에 대한 cost 값을 구한다.
            accur = self.accuracy(self.x, self.t)  # test data 에 대한 accur 값을 구한다.
            print("cost, accuracy : ", cost, ' ', accur)  # cost, accuracy 값을 출력한다.
            cost_list.append(cost)  # 그래프로 나타내기 위해 cost 값을 저장
            accuracy_list.append(accur) # 그래프로 나타내기 위해 accuracy 값을 저장
            grads = self.numerical_gradient(x_batch, t_batch)  # 편미분 값을 받아온다.
            self.params['W1'] -= lr * grads['W1']  # W1에 편미분 값을 lr 만큼 반영한다
            self.params['b1'] -= lr * grads['b1']  # b1에 편미분 값을 lr 만큼 반영한다
            self.params['W2'] -= lr * grads['W2']  # W2에 편미분 값을 lr 만큼 반영한다
            self.params['b2'] -= lr * grads['b2']  # b2에 편미분 값을 lr 만큼 반영한다

        training_accuracy = self.accuracy(x_batch, t_batch)
        print('Training Accuracy = ', training_accuracy)  # 학습을 마친 후 학습 test_data 에 대해서 최종 accuracy 구하여 출력
        cost_list = np.array(cost_list)  # list 를 numpy array 로 바꾼다
        accuracy_list = np.array(accuracy_list)  # list 를 numpy array 로 바꾼다
        plt.plot(np.arange(0, epoch), cost_list, label="loss")  # cost 값을 그래프로 표시한다.
        plt.plot(np.arange(0, epoch), accuracy_list, label="training accuracy")  # cost 값을 그래프로 표시한다.
        plt.xlabel("epoch")  # x축의 이름 설정
        plt.legend()
        plt.show()  # 그래프를 보여준다.

