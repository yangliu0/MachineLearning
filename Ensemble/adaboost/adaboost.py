# -*- coding: utf-8 -*-
# @Time    : 2018/11/7 19:37
# @Author  : Yang Liu

# simple AdaBoost

import numpy as np


class AdaBoost(object):
    def __init__(self, base_model='stump', verbose=0, num_boosting=40):
        """
        :param base_model: 基分类器
            stump: 单层决策树
        :param verbose: 是否打印中间学习过程 0:不打印 1:打印
        :param num_boosting: 迭代轮数
        """
        self.base_model = base_model
        self.verbose = verbose
        self.num_boosting = num_boosting

    def fit(self, X, y):
        """
        :param X: 数据集大小(m, n)
        :param y: 测试集大小 (m, 1)
        :return: model
        """
        weak_class_arr = []
        m = X.shape[0]
        D = np.ones((m, 1)) / m
        agg_class_est = np.zeros((m, 1))
        for i in range(self.num_boosting):
            if self.base_model == 'stump':
                best_stump, error, class_est = self._build_stump(X, y, D)
                alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-6)))
                best_stump['alpha'] = alpha
                weak_class_arr.append(best_stump)
                if self.verbose == 1:
                    print('class_est.T: ', class_est.T)
                expon = np.multiply(np.multiply(-1 * alpha, np.mat(y)), class_est)
                D = np.multiply(D, np.exp(expon))
                D = D / D.sum()
                agg_class_est += alpha * class_est
                if self.verbose == 1:
                    print('agg_class_est.T: ', agg_class_est.T)
                agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(y), np.ones((m, 1)))
                error_rate = agg_errors.sum() / m
                if self.verbose == 1:
                    print('total error: ', error_rate, '\n')
                if error_rate == 0.0:
                    break

        self.model = weak_class_arr
        return weak_class_arr

    def _build_stump(self, X, y, D):
        """
        :param X: 训练数据
        :param y: label
        :param D: 数据分布，包含对每个样本的权重
        :return: (best_stump, min_error, best_class_est)
            best_stump: dict 保存最佳分割树桩点
            min_error: 最小的分类错误率
            best_class_est: 最佳分类结果
        """

        (m, n) = X.shape
        num_steps = 10.0
        best_stump = {}
        best_class_est = np.zeros((m, 1))
        min_error = float('inf')
        for i in range(n):
            range_min = X[:, i].min()
            range_max = X[:, i].max()
            step_size = (range_max - range_min) / num_steps
            for j in range(-1, int(num_steps) + 1):
                for inequal in ['lt', 'gt']:
                    thresh_val = range_min + float(j) * step_size
                    predicted_vals = self._stump_classify(X, i, thresh_val, inequal)
                    err_arr = np.ones((m, 1))
                    err_arr[predicted_vals == y] = 0

                    weighted_error = float(np.dot(D.T, err_arr))

                    if self.verbose == 1:
                        print('split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f' \
                              % (i, thresh_val, inequal, weighted_error))

                    if weighted_error < min_error:
                        min_error = weighted_error
                        best_class_est = predicted_vals.copy()
                        best_stump['dim'] = i
                        best_stump['thresh'] = thresh_val
                        best_stump['ineq'] = inequal

        return best_stump, min_error, best_class_est

    def _stump_classify(self, X, dim, thresh_val, thresh_ineq):
        """
        :param X: 训练数据
        :param dim: 选择划分的维度
        :param thresh_val: 划分阈值
        :param thresh_ineq: 划分符号
            lt: 小于等于thresh_val的数据预测为-1
            gt: 大于thresh_val的数据预测为-1
        :return: 该划分情况下的预测结果
        """

        ret = np.ones((X.shape[0], 1))
        if thresh_ineq == 'lt':
            ret[X[:, dim] <= thresh_val] = -1
        else:
            ret[X[:, dim] > thresh_val] = -1
        return ret

    def predict(self, X):
        """
        预测函数
        :param X: 测试集
        :return: 预测的label
        """
        m = X.shape[0]
        agg_class_est = np.zeros((m, 1))
        for i in range(len(self.model)):
            if self.base_model == 'stump':
                class_est = self._stump_classify(X, self.model[i]['dim'], self.model[i]['thresh'],
                                                 self.model[i]['ineq'])
                agg_class_est += self.model[i]['alpha'] * class_est
                if self.verbose == 1:
                    print(agg_class_est)

        return np.sign(agg_class_est)
