import matplotlib.pyplot as plt
from numpy.ma import arange


# 描绘最佳拟合直线
def plotBestFit(w, b, dataMat, labelMat):
    m = dataMat.shape[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(m):
        if int(labelMat[i]) == 1:
            xcord1.append(dataMat[i, 0])
            ycord1.append(dataMat[i, 1])
        else:
            xcord2.append(dataMat[i, 0])
            ycord2.append(dataMat[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='x')
    ax.scatter(xcord2, ycord2, s=30, c='green', marker='o')
    x = arange(-3.0, 3.0, 0.1)
    y = (-w[0] * x - b) / w[1]
    ax.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
