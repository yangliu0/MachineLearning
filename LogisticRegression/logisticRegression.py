# -*- coding:utf-8 -*-

import numpy as np
from plotBestFit import plotBestFit


# sigmoid函数
def sigmoid(z):
    s = 1.0 / (1 + np.exp(-z))
    return s


# 计算梯度和损失值
def calculate_grads_cost(w, b, X, Y):
    m = X.shape[0]

    A = sigmoid(np.dot(X, w) + b)
    cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m

    dw = np.dot(X.T, A - Y)
    db = np.sum(A - Y)

    grads = {
        "dw": dw,
        "db": db
    }

    return grads, cost


# 使用梯度下降优化cost
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):
        grads, cost = calculate_grads_cost(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("第 %d 轮迭代后的损失值为： %f" % (i, cost))

    params = {
        "w": w,
        "b": b
    }

    grads = {
        "dw": dw,
        "db": db
    }

    return params, grads, costs


if __name__ == "__main__":
    data = np.loadtxt("testSet.txt")
    dataMat = data[:, 0:2]
    labelMat = data[:, 2]
    m, n = dataMat.shape
    labelMat = labelMat.reshape(m, 1)
    w = np.zeros((n, 1))
    b = 0
    params, grads, costs = optimize(w, b, dataMat, labelMat, 200, 0.1)

    w = params["w"]
    b = params["b"]
    # print(w)
    # print(b)

    # 可视化结果
    plotBestFit(w, b, dataMat, labelMat)
