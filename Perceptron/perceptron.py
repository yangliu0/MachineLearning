#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
author: Yang Liu

The primitive form of the perceptron
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def loadData():
    data = np.loadtxt('testSet.txt')
    dataMat = data[:, 0:2]
    labelMat = data[:, 2]
    return dataMat, labelMat


def sign(val):
    if val >= 0:
        return 1
    else:
        return -1


"""
    训练模型
    eta: learning rate
"""
def trainPerceptron(dataMat, labelMat, eta):
    m, n = dataMat.shape
    weight = np.zeros(n)
    bias = 0

    flag = True
    while flag:
        for i in range(m):
            if np.any(labelMat[i] * (np.dot(weight, dataMat[i]) + bias) <= 0):
                weight = weight + eta * labelMat[i] * dataMat[i].T
                bias = bias + eta * labelMat[i]
                print("weight, bias: ", end="")
                print(weight, end="  ")
                print(bias)
                flag = True
                break
            else:
                flag = False

    return weight, bias


# 可视化展示分类结果
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
    weight, bias = trainPerceptron(dataMat, labelMat, 1)
    plotResult(dataMat, labelMat, weight, bias)
