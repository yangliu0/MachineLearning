# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
author:Yang Liu

The dual form of perceptron
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def loadData():
    data = np.loadtxt('testSet.txt')
    dataMat = data[:, 0:2]
    labelMat = data[:, 2]
    labelMat = labelMat.reshape((labelMat.shape[0], 1))
    return dataMat, labelMat


"""
训练模型
b:bias
eta:learning rate
"""
def trainModel(dataMat, labelMat, alpha, b, eta):
    flag = True
    while flag:
        for i in range(m):
            if (labelMat[i, 0] * (np.sum((alpha * labelMat * np.dot(dataMat, dataMat[i].T).reshape((m, 1)))) + b)) <= 0:
                alpha[i] = alpha[i] + eta
                b = b + eta * labelMat[i]
                flag = True
                break
            else:
                flag = False
    alpha_new = alpha * labelMat
    w = np.dot(dataMat.T, alpha_new)
    return w, b


# 可视化结果
def plotResult(dataMat, labelMat, weight, bias):
    fig = plt.figure()
    axes = fig.add_subplot(111)

    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    for i in range(len(labelMat)):
        if (labelMat[i] == -1):
            type1_x.append(dataMat[i][0])
            type1_y.append(dataMat[i][1])

        if (labelMat[i] == 1):
            type2_x.append(dataMat[i][0])
            type2_y.append(dataMat[i][1])

    type1 = axes.scatter(type1_x, type1_y, marker='x', s=20, c='red')
    type2 = axes.scatter(type2_x, type2_y, marker='o', s=20, c='blue')

    y = (0.1 * -weight[0] / weight[1] + -bias / weight[1], 4.0 * -weight[0] / weight[1] + -bias / weight[1])
    axes.add_line(Line2D((0.1, 4.0), y, linewidth=1, color='blue'))

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()


if __name__ == "__main__":
    dataMat, labelMat = loadData()
    m, n = dataMat.shape
    alpha = np.zeros((m, 1))
    b = 0
    eta = 1
    w, b = trainModel(dataMat, labelMat, alpha, b, eta)
    print("w: ", end="")
    print(w)
    print("b: %d" % b)
    plotResult(dataMat, labelMat, w, b)
