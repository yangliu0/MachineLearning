from numpy import *
import matplotlib.pyplot as plt


# 加载数据
def loadDataSet():
    dataMat = []
    labelMat = []
    file = open('testSet.txt')
    for line in file.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


# sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 梯度上升算法
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    theta = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * theta)
        error = (labelMat - h)
        theta = theta + alpha * dataMatrix.transpose() * error
    return theta

# 描绘最佳拟合直线
def plotBestFit(theta):
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-theta[0]-theta[1]*x)/theta[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
