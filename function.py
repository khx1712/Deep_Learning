import numpy as np
import matplotlib.pyplot as plt

eMin = -np.log(np.finfo(type(0.1)).max)  # 709.78 최대값 설정

def step_function(x):
    return np.array(x > 0, dtype=np.int)


# 지수값이 너무 올라가면 max 를 초과하므로 한계를 설정한다.
def sigmoid(z):
    zSafe = np.array(np.maximum(z, eMin))
    return 1.0 / (1 + np.exp(-zSafe))


def reLU(x):
    return (np.maximum(x, 0))


def softmax(x):
    exp_a = np.exp(x-np.max(x))
    sum_exp_a = np.sum(exp_a)
    return exp_a/sum_exp_a



'''
x1 = np.array([1, -1, 800, -5, 1000])
x2 = np.array([1, -1, 5, 4])
x3 = np.array([1, -1, 800, -5, 1000, 9990])

print(sigmoid(x1))

x = np.arange(-5, 5, 0.1)
y = sigmoid(x)

plt.plot(x,y)
plt.show()

print(reLU(x1))

x = np.arange(-5, 5, 0.1)
y = reLU(x)

plt.plot(x, y)
plt.show()

x2 = np.array([1, -1, 5, 4])
x3 = np.array([1, -1, 800, -5, 1000, 9990])


s = softmax(x2)
print(np.sum(s))

s = softmax(x3)
print(np.sum(s))
'''