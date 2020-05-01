import numpy as np
from sklearn.datasets import load_iris
from knn import KNN

iris = load_iris()  # scikit-learn 에서 iris data 를 불러와 저장

X = iris.data  # 150*4
y = iris.target  # 150*1
y_name = iris.target_names  # target 이름 저장

cycle = 15  # 얼만큼의 주기로 test data 를 만들껀지 정한다.
for_test = np.array([(i % cycle == (cycle - 1)) for i in range(y.shape[0])])
# y.shape[0] 개수 150 / 주기에 해당하는 지점만 1으로 초기화해서 test array 를 만든다
for_train = ~for_test  # test 에 해당하는 지점을 제외한 점들을 모두 1로 만든다.

X_train = X[for_train]  # train 에 해당하는 지점의 값들으로 list 만든다
y_train = y[for_train]  # train 에 해당하는 지점의 값들으로 list 만든다

X_test = X[for_test]  # test 에 해당하는 지점의 값들으로 list 만든다
y_test = y[for_test]  # test 에 해당하는 지점의 값들으로 list 만든다

K_list = [3, 5, 10, 13, 15, 17, 19, 21, 23, 25, 27, 29]
# underfitting / overfitting 모두 확인하기 위해 K의 값을 (3~29) 더 넣는다.

for K in K_list:  # list 에 있는 K값 모두에 대해서 test 를 진행한다

    knn_iris = KNN(K, X_train, y_train, y_name)  # K, train 값들과 target 의 이름을 넘겨 KNN class 를 생성한다.
    print("\n-------------------- K =", K, "--------------------")
    print("\n< Majority vote >")
    true_count = 0  # 몇 퍼센트의 정확도를 가지는지 확인을 위한 count 변수
    for i in range(y_test.shape[0]):
        knn_iris.get_nearest_k(X_test[i])  # 가까운 점들은 어떤 점이고 거리가 어떻게 되는지 내부적으로 저장
        res = knn_iris.get_majority_vote()  # target 중에 무엇인지 y 값으로 준다
        print("Test Data: ", i, " Computed class: ", res, ",\tTrue class: ", y_name[y_test[i]])
        if res == y_name[y_test[i]]:  # 올바르면 count
            true_count += 1
        knn_iris.reset()
    print("correct : ", round((true_count * 100) / y_test.shape[0], 2), '%')  # 정학도가 몇 퍼센트인지 출력

    true_count = 0  # 몇 퍼센트의 정확도를 가지는지 확인을 위한 count 변수
    print("\n< Weighted majority vote >")
    for i in range(y_test.shape[0]):
        knn_iris.get_nearest_k(X_test[i])  # 가까운 점들은 어떤 점이고 거리가 어떻게 되는지 내부적으로 저장
        res = knn_iris.weighted_majority_vote()  # target 중에 무엇인지 y 값으로 준다
        print("Test Data: ", i, " Computed class: ", res, ",\tTrue class: ", y_name[y_test[i]])
        if res == y_name[y_test[i]]:  # 올바르면 count
            true_count += 1
        knn_iris.reset()
    print("correct : ", round((true_count * 100) / y_test.shape[0], 2), '%')  # 정학도가 몇 퍼센트인지 출력
