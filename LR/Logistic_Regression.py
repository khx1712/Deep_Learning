import numpy as np
from function import sigmoid, softmax
import sys
import matplotlib.pyplot as plt


class LogisticRegression:  # Multinomial Classification 을 위한 class

    def __init__(self, X_train, y_train, y_name):  # 초기화
        self.X = X_train  # test 값을 추정하기 위한 입력 값
        self.y = y_train  # test 값을 추정하기 위한 실제 결과 값
        self.y_name = y_name  # result 종류가 몇개인지 이름을 담고있다.
        self.W = np.random.randn(X_train.shape[1], y_name.shape[0])  # weight 값을(n,t)(feature*target 수)만큼 random 수로 초기화

    def cost(self, hx, y):  # cost 함수 계산 hx(m*t), y(m*t)
        m = y.shape[0]  # m 즉 data 의 개수를 저장
        # RuntimeWarning: divide by zero encountered in log 를 피하기 위해서 0을 아주작은 epsilon 값으로 바꿔준다
        hx = np.where(hx == 0, sys.float_info.epsilon, hx)  # hx 에서 0을 찾아서 epsilon 으로 바꿔준다
        one_hx = 1-hx
        one_hx = np.where(one_hx == 0, sys.float_info.epsilon, one_hx)  # one_hx 에서 0을 찾아서 epsilon 으로 바꿔준다
        cost = -np.sum(y*np.log(hx) + (1-y)*np.log(one_hx), axis=0)/m  # m*t 벡터를 sum 을 통해 m을 모두 더하여 (t,)로 합쳐준다.
        return cost

    def learn(self, epoch, learning_rate):  # epoch 학습 반복횟수, learning rate 를 입력받아 초기화된 train data 로 학습시킨다
        cost_list = list()  # cost 변화 그래프에 값을 주기 위해 cost 를 저장하는 list
        for epo in range(epoch):  # epoch 만큼 반복하여 학습한다.
            z = np.dot(self.X, self.W)  # m*t(data*label 수)개의 (W^T)*X
            hx = sigmoid(z)  # (m,t) data, 예측 값을 sigmoid 연산을 통해 0~1 사이의 값으로 변환한다.
            for i in range(self.W.shape[0]):  # n(feature)번 반복문 돌며 각 weight 에 대하여 cost 를 편미분한 값을 빼서 학습한다.
                xj = self.X[:, i]  # (m,) 편미분을 위해 j번째 x값을 data 수 만큼 잘라가져온다.
                # xxj = xj.reshape(xj.shape[0], 1)  # (113,1)
                # hx-self.y (m,t) -> (t, m) transpose 하여 (m,) 과 dot 연산을 통해 (t,) data 로 합쳐 편미분 값을 구하여 빼준다.
                self.W[i] -= learning_rate*(np.dot(np.transpose(hx-self.y), xj))
            cost_list.append(list(self.cost(hx, self.y)))  # cost 값을 list 에 추가해준다.
            print("epoch: ", epo, " cost: ", cost_list[epo])  # 몇번째 학습인지, cost 는 몇인지 매 학습 때 마다 출력한다.
        cost_list = np.array(cost_list)

        for i in range(cost_list.shape[1]):  # 10개의 target 에 대한 cost 값을 그래프로 표시한다.
            plt.plot(np.arange(0, epoch), cost_list[:,i], label=("target class " + str(i)))
        plt.xlabel("number of iterations")  # x축의 이름 설정
        plt.ylabel("cost")  # y축의 이름 설정
        plt.legend()
        plt.show()  # 그래프를 보여준다.

    def predict(self, X_test, y_test):  # 새로운 input 들어오면 예측
        z = np.dot(X_test, self.W)  # m(data 수)개의 (W^T)*X
        hx = sigmoid(z)  # (m,t) data, 예측 값을 sigmoid 연산을 통해 0~1 사이의 값으로 변환한다.
        sx = np.array([softmax(hx[i]) for i in range(hx.shape[0])])  # 예측값을 softmax 함수를 통해 확률값으로 바꿔준다.

        cnt = 0  # 실험값이 실제값과 얼마나 같은지 개수를 저장하는 변수
        for i in range(sx.shape[0]):  # test data 개수 만큼 반복한다.
            max_index = np.argmax(sx[i])  # 가장 큰 확률의 index 가 몇번째 인지 찾는다.
            if y_test[i][max_index] == 1:  # index 와 target 이 같은지 확인한다
                cnt += 1  # 같으면 1증가
        print("Accuracy: ", cnt / hx.shape[0])  # 몇퍼센트 만큼 적중에 성공했는지 출력


class LogisticRegressionBinary:  # Binary Classification 을 위한 class

    def __init__(self, X_train, y_train, y_name):  # 초기화
        self.X = X_train  # test 값을 추정하기 위한 입력 값
        self.y = y_train  # test 값을 추정하기 위한 실제 결과 값
        self.y_name = y_name  # result 종류가 몇개인지 이름을 담고있다.
        self.W = np.random.randn(X_train.shape[1], )  # weight 값을(n,)(feature 수)만큼 random 수로 초기화

    def cost(self, hx, y):  # cost 함수, 계산 hx(m,), y(m,)
        m = y.shape[0]  # m 즉 data 의 개수를 저장
        # RuntimeWarning: divide by zero encountered in log 를 피하기 위해서 0을 아주작은 epsilon 값으로 바꿔준다
        hx = np.where(hx == 0, sys.float_info.epsilon, hx)  # hx 에서 0을 찾아서 epsilon 으로 바꿔준다
        one_hx = 1-hx
        one_hx = np.where(one_hx == 0, sys.float_info.epsilon, one_hx)  # one_hx 에서 0을 찾아서 epsilon 으로 바꿔준다
        cost = -np.sum(y*np.log(hx) + (1 - y)*np.log(one_hx))/m  # (m,) 를 sum 을 통해 모두 더하여준다.
        return cost

    def learn(self, epoch, learning_rate):  # epoch 학습 반복횟수, learning rate 를 입력받아 초기화된 train data 로 학습시킨다
        cost_list = list()  # cost 변화 그래프에 값을 주기 위해 cost 를 저장하는 list
        for epo in range(epoch):  # epoch 만큼 반복하여 학습한다.
            z = np.dot(self.X, self.W)  # m(data 수)개의 (W^T)*X
            hx = sigmoid(z)  # (m,) data, 예측 값을 sigmoid 연산을 통해 0~1 사이의 값으로 변환한다.
            for i in range(self.W.shape[0]):  # n(feature)번 반복문 돌며 각 weight 에 대하여 cost 를 편미분한 값을 빼서 학습한다.
                xj = self.X[:, i]  # (m,) 편미분을 위해 j번째 x값을 data 수 만큼 잘라가져온다.
                # hx-self.y (m,) 과 xj를 dot 연산을 통해 모두 더하여 편미분 값을 구하여 빼준다.
                self.W[i] -= learning_rate*(np.dot((hx-self.y), xj))
            cost_list.append(self.cost(hx, self.y))  # cost 값을 list 에 추가해준다.
            print("epoch: ", epo, " cost: ", cost_list[epo])  # 몇번째 학습인지, cost 는 몇인지 매 학습 때 마다 출력한다.
        cost_list = np.array(cost_list)
        plt.plot(np.arange(0, epoch), cost_list, label="binary class")  # cost 값을 그래프로 표시한다.
        plt.xlabel("number of iterations")  # x축의 이름 설정
        plt.ylabel("cost")  # y축의 이름 설정
        plt.legend()
        plt.show()  # 그래프를 보여준다.

    def predict(self, X_test, y_test):  # 새로운 input 들어오면 예측
        z = np.dot(X_test, self.W)  # m(data 수)개의 (W^T)*X
        hx = sigmoid(z)  # (m,) data, 예측 값을 sigmoid 연산을 통해 0~1 사이의 값으로 변환한다.
        cnt = 0  # 실험값이 실제값과 얼마나 같은지 개수를 저장하는 변수
        for i in range(hx.shape[0]):
            # target 이 맞은걸 맞다고 예측하거나, 틀린걸 틀리다고 예측하였는지 실제 값과 비교한다.
            if (hx[i] > 0.5 and y_test[i] == 1) or (hx[i] <= 0.5 and y_test[i] == 0):
                cnt += 1  # 맞으면 1증가
        print("Accuracy: ", cnt/hx.shape[0])  # 몇퍼센트 만큼 적중에 성공했는지 출력
