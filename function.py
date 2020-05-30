import numpy as np
import matplotlib.pyplot as plt
import sys

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
    exp_a = np.exp(x - (x.max(axis=1).reshape([-1, 1])))
    exp_a /= exp_a.sum(axis=1).reshape([-1, 1])
    return exp_a

def cross_entropy_error_num(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def cross_entropy_error_onehot(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t*np.log(y + 1e-7)) / batch_size


def f2(x):
    y=0
    for i in range(x.shape[0]):
        y += x[i]**2
    return y


def numerical_gradient(f, x):  # x =
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in np.ndindex(x.shape):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)
        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)

        grad[idx] = (fxh1-fxh2)/(2*h)
        x[idx] = tmp_val
    return grad

# print(numerical_gradient(f2, np.array([3.0, 4.0])))


def gradient_descent(f, init_x, lr=0.01, epoch=100):
    x = init_x

    for i in range(epoch):
        x -= lr * numerical_gradient(f, x)
    return x

'''
print(gradient_descent(f2, np.array([3.0, 4.0]), lr=10, epoch=100))
print(gradient_descent(f2, np.array([3.0, 4.0]), lr=10, epoch=10))
print(gradient_descent(f2, np.array([3.0, 4.0]), lr=1e-5, epoch=100))
print(gradient_descent(f2, np.array([3.0, 4.0]), lr=1e-3, epoch=100))
print(gradient_descent(f2, np.array([3.0, 4.0]), lr=1e-3, epoch=10000))
print(gradient_descent(f2, np.array([3.0, 4.0]), lr=1e-3, epoch=100000))
print(gradient_descent(f2, np.array([3.0, 4.0]), lr=1e-1, epoch=1000))
'''