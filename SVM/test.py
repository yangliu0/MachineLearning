# -*- coding: utf-8 -*-
# @Time    : 2018/11/2 16:41
# @Author  : Yang Liu

# 测试函数

from SVM_SMO import SVMSMO
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    testSet = pd.read_csv('dataset/testSet.txt', sep='\t', header=None)
    testSet.columns = ['x1', 'x2', 'label']
    y = testSet.pop('label')
    X = testSet

    model = SVMSMO()

    support_vectors = model.fit(X.values, y.values)
    print(support_vectors)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x_cord0 = []
    y_cord0 = []
    x_cord1 = []
    y_cord1 = []
    for index, x in enumerate(X.values):
        if y.values[index] == -1:
            x_cord0.append(x[0])
            y_cord0.append(x[1])
        else:
            x_cord1.append(x[0])
            y_cord1.append(x[1])

    ax.scatter(x_cord0, y_cord0, marker='s', s=90)
    ax.scatter(x_cord1, y_cord1, marker='o', s=50, c='red')

    for vec in support_vectors:
        circle = Circle((vec[0], vec[1]), 0.5, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
        ax.add_patch(circle)

    plt.show()


if __name__ == '__main__':
    main()
