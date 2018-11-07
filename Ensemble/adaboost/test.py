# -*- coding: utf-8 -*-
# @Time    : 2018/11/7 20:34
# @Author  : Yang Liu

# 测试函数
from adaboost import AdaBoost
import numpy as np


def main():
    X_train = np.array([
        [1.0, 2.1],
        [2.0, 1.1],
        [1.3, 1.0],
        [1.0, 1.0],
        [2.0, 1.0]
    ])

    y_train = np.array([1.0, 1.0, -1.0, -1.0, 1.0]).reshape((-1, 1))
    model = AdaBoost(verbose=1)
    model.fit(X_train, y_train)

    X_test = np.array([
        [5, 5],
        [0, 0]
    ])

    y_predict = model.predict(X_test)
    print('predict result: ', y_predict)

if __name__ == '__main__':
    main()
