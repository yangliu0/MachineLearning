# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    KNN algorithm

    author:Yang Liu

    use KNN algorithm to achieve digital recognition
"""

import numpy as np
from os import listdir


# 读取32*32图像文本,转为向量返回
def img2vector(filename):
    retVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            retVect[0, 32 * i + j] = int(lineStr[j])
    return retVect


# 预处理训练集和数据集，将图片文本向量化
def preprocessing():
    # 训练集
    trainFileList = listdir("trainingData/")
    m_train = len(trainFileList)
    trainData = np.zeros((m_train, 1024))
    trainLabel = []
    for i in range(m_train):
        trainData[i, :] = img2vector("trainingData/" + trainFileList[i])
        trainLabel.append(int(trainFileList[i].split('_')[0]))
    trainLabel = np.array(trainLabel)

    # 测试集
    testFileList = listdir("testData/")
    m_test = len(testFileList)
    testData = np.zeros((m_test, 1024))
    testLabel = []
    for i in range(m_test):
        testData[i, :] = img2vector("testData/" + testFileList[i])
        testLabel.append(int(testFileList[i].split('_')[0]))
    testLabel = np.array(testLabel)
    return trainData, trainLabel, testData, testLabel


# kNN算法
def kNN(testData, trainDataMat, trainLabelMat, k):
    m = trainDataMat.shape[0]

    # 计算距离,使用欧式距离
    difMat = trainDataMat - testData
    sqDifMat = difMat ** 2
    sqDistance = sqDifMat.sum(axis=1)
    distance = sqDistance ** 0.5

    # 对距离向量进行排序，获得其排序后的index
    sortedDistanceIndicies = distance.argsort()

    # 对前k个label进行计数排序
    classCount = {}
    for i in range(k):
        voteLabel = trainLabelMat[sortedDistanceIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    # 按item进行排序逆序排序，第一个label即为选举出的label
    result = sorted(classCount.items(), key=lambda item: item[1], reverse=True)
    return result[0][0]


if __name__ == "__main__":
    # 获取训练集和测试集的向量
    trainDataMat, trainLabelMat, testDataMat, testLabelMat = preprocessing()

    m = testDataMat.shape[0]
    count = 0
    # 预测过程
    for i in range(m):
        predict = kNN(testDataMat[i, :], trainDataMat, trainLabelMat, 5)
        if predict != testLabelMat[i]:
            count += 1
        print("预测结果为: %d, 期望结果为: %d" % (predict, testLabelMat[i]))

    print("预测错误率为: %.2f %%" % (count / m * 100))
