import math
import numpy as np


class KNN:
    nearest = []  # 인접값을 내부적으로 저장하는 list

    def __init__(self, K, X_train, y_train, y_name):  # 생성자로 값을 받으면 초기화 되어 object 가 생성된다
        self.K = K  # 인접점을 몇개까지 반영할 것인지
        self.X = X_train  # test 값을 추정하기 위한 입력 값
        self.y = y_train  # test 값을 추정하기 위한 실제 결과 값
        self.y_name = y_name  # result 종류가 몇개인지 이름을 담고있다.

    def cal_distance(self, x1, x2):  # 두점을 입력받아 두점사이의 distance 를 return 해준다
        sum = 0
        for i in range(x1.shape[0]):
            sum += np.power(x1[i] - x2[i], 2)
        return np.sqrt(sum)

    def get_nearest_k(self, x_test):  # 입력받은 x 값에서 가까운 k개의 점들을 찾아 nearest 에 저장해준다
        res = []
        for i in range(self.X.shape[0]):
            res.append([self.cal_distance(self.X[i], x_test), i])  # 가까운 값들이 어떤점인지 알기 위해  [distance, index] list 를 만든다.
        res.sort()  # 가까운 점을 찾기 위해 distance 가 작은 순서로 정렬해준다
        self.nearest = res[:self.K]  # K개 만큼의 점을 반영하기 위해 K 개 만큼 slice 하여 nearest 에 저장한다

    def reset(self):  # 초기화해준다
        self.nearest = []

    def get_majority_vote(self):  # 가중치를 반영하지 않고 어떤 결과의 점이 가장 많은지 return 해준다
        count = [0] * len(self.y_name)  # 뭐가 가장많은지 count 위해 0으로 초기화된 list 를 만든다
        for i in range(len(self.nearest)):
            nearest = self.nearest[i][1]
            y_hat = self.y[nearest]  # 가까운 점의 결과 값(y)이 무엇인지 예상 값을 추정한다
            count[y_hat] += 1

        majority = 0
        for i in range(len(self.y_name)):  # 가장 많은 값이 무엇인지 찾아 majority 에 저장해준다
            if count[majority] < count[i]:
                majority = i

        return self.y_name[majority]  # majority 에 해당하는 이름을 return 해준다.

    def weighted_majority_vote(self):
        weight = list()  # 가중치 입력을 위한 비어있는 list 만든다
        weight_sum = 0
        for i in range(self.K):
            weight_sum += 1 / self.nearest[i][0]  # 값 환산을 위해 더해준다
        for i in range(self.K):
             weight.append([1 / self.nearest[i][0] / weight_sum, self.nearest[i][1]])  # 값 환산을 위해 더해준다

        weight_count = [0] * len(self.y_name)  # 가중치가 count 위한 list
        for i in range(len(weight)):
            nearest = weight[i][1]
            y_hat = self.y[nearest]
            weight_count[y_hat] += weight[i][0]

        majority = 0
        for i in range(len(self.y_name)):  # 가장 큰 가중치를 가지는 y값을 majority 에 저장
            if weight_count[majority] < weight_count[i]:
                majority = i

        return self.y_name[majority]  # majority 에 해당하는 이름을 return
