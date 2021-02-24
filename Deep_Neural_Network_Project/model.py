# coding: utf-8
# 2020/인공지능/B511118/오승찬
import sys
import os
from collections import OrderedDict
import pickle
import numpy as np
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정


def softmax(x):  # 확률값으로 바꿔준다.
    if x.ndim == 2:  # 2차원이면 한줄씩 계산하여 결과를 1차원으로 준다.
        x = x.T  # 행과 열을 바꿔준다.
        x = x - np.max(x, axis=0)  # 가장 큰값을 빼서 overflow 를 방지한다.

        y = np.exp(x) / np.sum(np.exp(x), axis=0)  # 전체의 합으로 나눠 확룰 값으로 바꿔준다.
        return y.T

    x = x - np.max(x)  # 가장 큰값을 빼서 overflow 를 방지한다.
    return np.exp(x) / np.sum(np.exp(x))  # 전체의 합으로 나눠 확룰 값으로 바꿔준다.


def cross_entropy_error(y, t):  # loss 값을 계싼해서 넘겨준다.
    if y.ndim == 1:  #
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class Relu:  # x<=0 -> 0 / x>0 -> x 으로 바꿔주는 switch 함수
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)  # 0보다 작은 값들을 mask 에 저장한다
        out = x.copy()  # x 값 임시저장
        out[self.mask]=0  # x 값중 mask 에 해당하는 값들 0으로 바꿔준다
        return out

    def backward(self, dout):
        dx = dout.copy()  # dout 값 임시저장
        dx[self.mask] = 0  # dout 값중 mask에 해당하는 값들을 0으로 바꿔준다.
        return dx


class BatchNormalization:  # 학습을 빨리하게 해준다/ 초기값에 의존하디 않는다/ 오버피팅을 억제한다. 3가지 기능의 layer
    def __init__(self, gamma, beta):  # forward 에서 계산에 쓰인 값이나 계산된 값을 backward 에 쓰기 위해 저장해놓는다.
        self.gamma = gamma
        self.beta = beta
        self.xmu = None
        self.var = None
        self.sqrtvar = None
        self.ivar = None
        self.xhat = None
        self.dG = None  # 미분값을 updata 에서 반영해주기 위해 저장
        self.dB = None  # 미분값을 updata 에서 반영해주기 위해 저장

    def forward(self, x):
        N, D = x.shape  # N: data의 개수, D: layer node 의 개수
        mu = 1. / N * np.sum(x, axis=0)  # x 값을 모두 더하여 N(data개수)로 나눠준다.
        self.xmu = x - mu  # 위의 값을 x에서 빼준다
        sq = self.xmu ** 2  # 위의 값을 제곱하여 저장해놓는다.
        self.var = 1. / N * np.sum(sq, axis=0)  # 제곱해 놓은 x값들을 모두 더하여 N(data개수)로 나눠준다.
        self.sqrtvar = np.sqrt(self.var + 1e-7)  # 위의 값에 epsilon 값을 더하여 제곱근을 구한다.
        self.ivar = 1. / self.sqrtvar  # 위의 값에 역수를 취한다.
        self.xhat = self.xmu * self.ivar  # 위의 값과 전에 구한 값(xmu)을 곱해준다.
        out = (self.gamma * self.xhat) + self.beta  # 위의 값을 처음에 초기화 해두었던 gamma, beta 값으로 계산해준다.
        return out

    def backward(self, dout):
        N,D = dout.shape  # N: data의 개수, D: layer node 의 개수
        self.dB = np.sum(dout, axis=0)  # dout 의 합은 Beta의 미분값이다.
        dgammax = dout
        self.dG = np.sum(dgammax*self.xhat, axis=0)  # dgrammax 와 xhat의 곱의 합은 Gamma의 미분값이다.
        dxhat = dgammax * self.gamma
        divar = np.sum(dxhat*self.xmu, axis=0)
        dxmu1 = dxhat * self.ivar
        dsqrtvar = -1. /(self.sqrtvar**2) * divar
        dvar = 0.5 * 1. /np.sqrt(self.var+1e-7) * dsqrtvar
        dsq = 1. /N * np.ones((N,D)) * dvar
        dxmu2 = 2 * self.xmu * dsq
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
        dx2 = 1. /N * np.ones((N,D)) * dmu
        dx = dx1 + dx2
        return dx


class Affine: # xW + b 를 (행열 연산을 해주는) 구해주는 layer 이다.
    def __init__(self, W, b):
        self.W = W  # weight 값 저장
        self.b = b  # bias 값 저장
        self.x = None
        self.dW = None  # weight 값의 미분값을 나중 update에 반영해주기 위해 저장해둘 변수
        self.db = None  # bias 값의 미분값을 나중 update에 반영해주기 위해 저장해둘 변수

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W)+self.b  # xW + b 값을 계산한다.
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)  # dout 과 W.T 를 행렬 연산해준다.
        self.dW = np.dot(self.x.T, dout)  # x.T와 dout 를 행렬 연산 해줘 weight의 미분값을 구한다.
        self.db = np.sum(dout, axis=0)  # dout 를 모두 더해줘 bias의 미분값을 구한다.
        return dx


class SoftmaxWithLoss:  # loss 값을 구해주는 layer 이다.
    def __init__(self):
        self.loss = None  # loss 값을 저장해두는 변수
        self.y = None  # backward 에서 사용할 y값을 저장해 두는 변수
        self.t = None  # backward 에서 사용할 t값을 저장해 두는 변수

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)  # x 값을 0~1 사이의 값으로 바꿔준다.
        self.loss = cross_entropy_error(self.y, self.t)  # loss 값을 계산해준다.
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]  # data 의 개수를 통해 batch size 를 구한다.
        dx = (self.y-self.t)/batch_size
        return dx


class SGD:  # 확률적경사하강법 optimizer 함수이다.
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):  # parameter에 변화값을 반영해준다.
        for key in params:
            params[key] -= self.lr * grads[key]  # learning rate를 반영한 변화값을 parameter에 반영해준다.


class AdaGrad:  # AdaGrad optimizer 함수로 변화율이 작아질수록 조금씩 반영해줘서 lr 를 낮춰준다.
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):  # parameter에 변화값을 반영해준다.
        if self.h is None:
            self.h = {}
            for key, val in params.items():  # parameter 들과 같은 모양으로 0으로 초기화 시켜 h를 만들어준다.
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key]*grads[key]  # 위에서 초기화 시킨 h에 변화율의 제곱을 더해준다
            params[key] -= self.lr * grads[key]/(np.sqrt(self.h[key])+1e-7)  # h 값까지 반영하여 parameter 를 수정해준다.


class Model:  # Neural Network Model 이다. input layer, 여러개의 hidden layer, output layer 로 이뤄져있다.
    def __init__(self, lr=0.01):
        self.params = {}  # parameter를 저장해주는 dictionary
        self.layers = OrderedDict()  # layer 가 순서를 유지할수 있는 dictionary를 생성
        self.__init_weight(6, 100, 100, 100, 100, 6)  # 각 layer의 node 개수를 받아 parameter를 초기화 시켜 params 에 담아준다.
        self.__init_layer()  # 여러 종류의 layer를 초기화 시켜 순서대로 layers 에 담아준다.
        self.optimizer = AdaGrad(lr)  # AdaGrad를 optimizer로 저장해준다.

    def __init_layer(self):  # layers 를 초기화 : 총 6개의 layer로 이뤄져 있고 각 layer를 여러개의 subLayer로 이뤄져있다.
        # input layer (Affine -> Batch Normalization -> Relu  순서로 연결되있다.)
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['BatchNorm1'] = BatchNormalization(self.params['G1'], self.params['B1'])
        self.layers['Relu1'] = Relu()

        # hidden layer 1 (Affine -> Batch Normalization -> Relu  순서로 연결되있다.)
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['BatchNorm2'] = BatchNormalization(self.params['G2'], self.params['B2'])
        self.layers['Relu2'] = Relu()

        # hidden layer 2 (Affine -> Batch Normalization -> Relu  순서로 연결되있다.)
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['BatchNorm3'] = BatchNormalization(self.params['G3'], self.params['B3'])
        self.layers['Relu3'] = Relu()

        # hidden layer 3 (Affine -> Batch Normalization -> Relu  순서로 연결되있다.)
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['BatchNorm4'] = BatchNormalization(self.params['G4'], self.params['B4'])
        self.layers['Relu4'] = Relu()

        # output layer (Affine -> Softmax -> Cross Entropy Error 순서로 연결되 있다.)
        self.layers['Affine5'] = Affine(self.params['W5'], self.params['b5'])
        self.last_layer = SoftmaxWithLoss()

    # Weight 를 포함한 params 를 초기화 시켜준다 : 각 layer 사이의 weight, bias, Gamma, Beta 을 초기화 시켜준다.
    def __init_weight(self, input_size, hidden1_size, hidden2_size, hidden3_size, hidden4_size, output_size):
        # input layer -> hidden layer 1
        self.params['W1'] = np.random.randn(input_size, hidden1_size) / (2*np.sqrt(input_size))
        self.params['G1'] = np.ones(hidden1_size)
        self.params['B1'] = np.zeros(hidden1_size)
        self.params['b1'] = np.zeros(hidden1_size)

        # hidden layer 1 -> hidden layer 2
        self.params['W2'] = np.random.randn(hidden1_size, hidden2_size) / (2*np.sqrt(hidden1_size))
        self.params['G2'] = np.ones(hidden2_size)
        self.params['B2'] = np.zeros(hidden2_size)
        self.params['b2'] = np.zeros(hidden2_size)

        # hidden layer 2 -> hidden layer 3
        self.params['W3'] = np.random.randn(hidden2_size, hidden3_size) / (2*np.sqrt(hidden2_size))
        self.params['G3'] = np.ones(hidden3_size)
        self.params['B3'] = np.zeros(hidden3_size)
        self.params['b3'] = np.zeros(hidden3_size)

        # hidden layer 3 -> hidden layer 4
        self.params['W4'] = np.random.randn(hidden3_size, hidden4_size) / (2*np.sqrt(hidden3_size))
        self.params['G4'] = np.ones(hidden4_size)
        self.params['B4'] = np.zeros(hidden4_size)
        self.params['b4'] = np.zeros(hidden4_size)


        self.params['W5'] = np.random.randn(hidden4_size, output_size) / (2*np.sqrt(hidden4_size))
        self.params['b5'] = np.zeros(output_size)

    def update(self, x, t):  # gradient(변화율)을 받아 parameter에 반영해준다.
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구한 뒤
         옵티마이저 클래스를 사용해서 네트워크 파라미터를 업데이트 해주는 함수입니다.

        :param x: train_data
        :param t: test_data
        """
        grads = self.gradient(x, t)  # backward propagation 을 통해 각 parameter의 변화율을 구하여 받는다.
        self.optimizer.update(self.params, grads)  # parameter에 반영해준다.

    def predict(self, x):  # data를 입력받아 y값을 예측합니다.
        """
        데이터를 입력받아 정답을 예측하는 함수입니다.

        :param x: data
        :return: predicted answer
        """

        for layer in self.layers.values():
            x = layer.forward(x)  # forward propogation을 통해 output을 다음 layer의 input으로 넘기는 과정을 반복, predict 값을 구한다.
        return x

    def loss(self, x, t):  # x를 통해 예측한 값 y와 실제 결과인 t를 통해 손실값을 구한다.
        """
        데이터와 레이블을 입력받아 로스를 구하는 함수입니다.
        :param x: data
        :param t: data_label
        :return: loss
        """
        y = self.predict(x)  # x를 통해 예측값을 구한다.
        return self.last_layer.forward(y, t)  # y(예측값)와 t(실제값)의 손실값을 구하여 return 한다.


    def gradient(self, x, t):  # forward propagation 과 backwowd propagation을 통해 parameter의 미분값을 계산해준다.
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구하는 함수입니다.
        첫번째로 받은데이터를 forward propagation 시키고,
        두번째로 back propagation 시켜 grads에 미분값을 리턴합니다.
        :param x: data
        :param t: data_label
        :return: grads
        """
        # forward propagation
        self.loss(x, t)

        # backward propagation을 하여 parameter의 미분값을 구한다.
        dout = 1
        dout = self.last_layer.backward(dout)  # output -> hidden4 로 backword

        layers = list(self.layers.values())
        layers.reverse()  # layers 를 backword propagation 위해서 역순으로 바꿔준다.
        for layer in layers:  # hidden4 -> hidden3 -> hidden2 -> hidden1 -> input 순서로 backword 진행한다.
            dout = layer.backward(dout)

        # backword propagation을 통해 구한 parameter 들의 미분값 을 저장해준다.
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['G1'] = self.layers['BatchNorm1'].dG
        grads['B1'] = self.layers['BatchNorm1'].dB
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['G2'] = self.layers['BatchNorm2'].dG
        grads['B2'] = self.layers['BatchNorm2'].dB
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['G3'] = self.layers['BatchNorm3'].dG
        grads['B3'] = self.layers['BatchNorm3'].dB
        grads['b3'] = self.layers['Affine3'].db
        grads['W4'] = self.layers['Affine4'].dW
        grads['G4'] = self.layers['BatchNorm4'].dG
        grads['B4'] = self.layers['BatchNorm4'].dB
        grads['b4'] = self.layers['Affine4'].db
        grads['W5'] = self.layers['Affine5'].dW
        grads['b5'] = self.layers['Affine5'].db
        return grads

    def save_params(self, file_name="params.pkl"):  # 학습을 진행한 parameter 값과 layer를 params.pkl 에 저장 시켜준다.
        """
        네트워크 파라미터를 피클 파일로 저장하는 함수입니다.

        :param file_name: 파라미터를 저장할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        params = {}
        layers = {}
        for key, val in self.params.items():  # 모든 parameter 저장
            params[key] = val
        for key, val in self.layers.items():  # 모든 layer 저장
            layers[key] = val
        with open(file_name, 'wb') as f:  # params.pkl를 write 권한으로 연다.
            pickle.dump(params, f)  # params.pkl 에 parameter를 써준다.
            pickle.dump(layers, f)  # params.pkl 에 layer를 써준다.

    def load_params(self, file_name="params.pkl"):  # 학습을 하여 저장해 놓은 parameter과 layer를 불러온다.
        """
        저장된 파라미터를 읽어와 네트워크에 탑재하는 함수입니다.

        :param file_name: 파라미터를 로드할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        with open(file_name, 'rb') as f:  # params.pkl를 read 권한으로 연다.
            params = pickle.load(f)  # params.pkl 에서 parameter를 읽어온다.
            layers = pickle.load(f)  # params.pkl 에서 layer를 읽어온다.
        for key, val in params.items():
            self.params[key] = val  # parameter를 하나씩 저장
        for key, val in layers.items():
            self.layers[key] = val  # layer를 하나씩 저장
