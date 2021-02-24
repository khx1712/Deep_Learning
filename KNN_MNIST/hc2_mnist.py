import sys, os
sys.path.append(os.pardir)  # 현재 폴더의 부모 폴더에서 찾아서 loading
import numpy as np
from dataset.mnist import load_mnist
from KNN.knn import KNN


def hand_craft(old_X):
    new_X = list()  # 변환한 data를 담을 list를 초기화
    for n in range(old_X.shape[0]):  # data의 개수 만큼 반복한다.
        temp = list()  # 변환한 data를 담을 list를 초기화
        for i in range(0, old_X[n].shape[0], 4):  # feature 을 4개씩 나눠준다
            sum = 0  # 값들의 합을 구한다
            for j in range(4):
                sum += old_X[n][i + j]  # 실수 형태의 data를 실수로 받기위해 그대로 더한다
            temp.append(sum)
        new_X.append(temp)
    return np.array(new_X)  # list 를 numpy 형식으로 바꿔준다

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, flatten=True)

label_name = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

hc_x_train = hand_craft(x_train)  # 기존의 data 를 넣어 feature 이 변화된 형태의 data 를 받는다
hc_x_test = hand_craft(x_test)  # 기존의 data 를 넣어 feature 이 변화된 형태의 data 를 받는다



K_list = [3]  # feature 가 많기 때문에 시간이 오래걸려 K의 값을 3만 시뮬레이션 해본다 넣는다.
size = 50
sample = np.random.randint(0,t_test.shape[0] , size)

for K in K_list:  # list 에 있는 K값 모두에 대해서 test 를 진행한다

    knn_iris = KNN(K, hc_x_train, t_train, label_name)  # K, train 값들과 target 의 이름을 넘겨 KNN class 를 생성한다.
    print("\n-------------------- K =", K, "--------------------")

    true_count = 0  # 몇 퍼센트의 정확도를 가지는지 확인을 위한 count 변수
    print("\n< Weighted majority vote >")
    for i in sample:
        knn_iris.get_nearest_k(hc_x_test[i])  # 가까운 점들은 어떤 점이고 거리가 어떻게 되는지 내부적으로 저장
        res = knn_iris.weighted_majority_vote()  # target 중에 무엇인지 y 값으로 준다
        print("Test Data: ", i, " Computed class: ", res, ",\tTrue class: ", label_name[t_test[i]])
        if res == label_name[t_test[i]]:  # 올바르면 count
            true_count += 1
        knn_iris.reset()
    print("correct : ", round((true_count * 100) / size, 2), '%')  # 정학도가 몇 퍼센트인지 출력
