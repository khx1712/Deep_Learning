# K-Nearest Neighbor
KNN(K-Nearest Neighbor) 알고리즘은 분류나 회귀에서 사용되는 가장 직관적이고 간단한 알고리즘이지만 그것에 비해서 정확도가 높아 아직도 사용되는 알고리즘입니다.\
4개의 feture을 가지고 3종류(setosa, versicolor, virginica)의 결과를 가지는 150개의 iris data를 KNN을 이용하여 분류하였습니다.

-------
 150개의 data중에 index=14mod15에 해당하는 data를 test data로 잡고 나머지를 train data로 잡고 진행하였는데, test data에서 가까운 거리의 점을 몇 개까지 반영하는지를 나타내는 K값을 3, 5, 10에서 진행하여

+ K = 3, 5, 10
+ (1)가중치를 반영하지 않는 : majority vote
+ (2)가중치를 반영하는 : weighted majority vote

 위의 조건에 해당하는 총 6번의 시뮬래이션을 진행하였습니다.
 
_# 가중치를 반영하는데 있어서 주의해야 할 것은 가중치 계산 과정에 있어서 1/(distance) 값이 무한대가 될 수 있다는 것입니다._


![noname01](https://user-images.githubusercontent.com/42955392/108977196-a4c8e500-76cb-11eb-8205-117a8ec4c57c.png)
![noname02](https://user-images.githubusercontent.com/42955392/108977502-f2455200-76cb-11eb-9cfd-c9edb59ec406.png)\
 k=3 에 해당하는 시뮬레이션 결과를 본다면, 가중치를 반영한 결과와, 가중치를 반영하지 않은 결과 모두 Data 7번에서 실제 값인 virginica가 아니라 versicolor로 잘못된 결과가 나와 90%의 적중률을 얻을 수 있었습니다.
 이러한 결과를 보자면 k가 3으로 너무 작기 때문에 여러 data에 대한 정보를 추정할 수 없어서 underfitting돼 이런 결과가 나타나게 되었다고 생각합니다. 

 그렇다면 k를 5로 늘려 진행해보았습니다. k=5 또한 3일 때와 같이, 가중치를 반영한 결과와, 가중치를 반영하지 않은 결과 모두 Data 7번에서 실제 값인 virginica가 아니라 versicolor로 잘못된 결과가 나와 90%의 적중률을 얻을 수 있었습니다.
 이러한 결과를 보자면 k가 5으로 너무 작기 때문에 여러 data에 대한 정보를 추정할 수 없어서 underfitting돼 이런 결과가 나타나게 되었다고 생각합니다. 

![noname03](https://user-images.githubusercontent.com/42955392/108977592-0b4e0300-76cc-11eb-9e50-1681504373b7.png)\
k를 10으로 늘려 진행하여 보겠습니다. k=10 에서는 그전과 다르게, 가중치를 반영한 결과와, 가중치를 반영하지 않은 결과 모두 그전에 오류가 생겼던 Data 7번에서 옳은 결과가 나와 100%의 적중률을 얻을 수 있었습니다. 이러한 결과를 보자면 k가 10으로  data에 대한 정보를 추정하기에 충분한 값이기 때문에 underfitting없이 좋은 결과가 나타나게 되었다고 생각합니다. 

그렇다면 k값을 무조건 늘린다고 좋은 것일까? 라는 생각에 k가 얼마 이상이 되야 overfitting이 생기는지  값을 2씩 늘려가며 확인해본 결과\
![noname04](https://user-images.githubusercontent.com/42955392/108977724-320c3980-76cc-11eb-839c-0c02b07f6125.png)
![noname06](https://user-images.githubusercontent.com/42955392/108977732-333d6680-76cc-11eb-9eed-9b25c141c61e.png)\
k가 25인 시점부터 overfitting이 생기는 것을 확인할 수 있었습니다.
