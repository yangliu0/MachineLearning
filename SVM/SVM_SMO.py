# -*- coding: utf-8 -*-
# @Time    : 2018/11/2 10:20
# @Author  : Yang Liu

# 使用SMO来实现SVM
# 简化版实现

import numpy as np
import random


class SVMSMO(object):
    def __init__(self, max_iter=1000, kernel_type='linear', C=1, epsilon=0.001, sigma=5.0):
        """
        :param max_iter: 最大迭代次数
        :param kernel_type: kernel的类型
                        'linear': 使用线性函数
                        'quadratic': 使用二次多项式kernel函数
                        'rbf': 使用径向基kernel函数
        :param C: 正则项value
        :param epsilon: 容忍度
        :param sigma: 针对rbf kernel function的参数
        """

        self.kernels = {
            'linear': self._kernel_linear,
            'quadratic': self._kernel_quadratic,
            'rbf': self._kernel_rbf
        }

        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon
        self.sigma = sigma

    def fit(self, X, y):
        """
        :param X: train set (m, n): m-训练数据量 n-数据维度
        :param y: 标签
        :return:
        """
        (m, n) = X.shape
        alpha = np.zeros((m, 1))
        kernel = self.kernels[self.kernel_type]
        iter = 0
        while iter < self.max_iter:
            alphaParisChanged = 0
            alpha_prev = np.copy(alpha)
            for i in range(m):
                # 计算w b
                self.w = self._cacl_w(X, y, alpha)
                self.b = self._cacl_b(X, y, self.w)
                X_i, y_i = X[i, :], y[i]
                # 计算error
                E_i = self._E(X_i, y_i, self.w, self.b)
                if ((y_i * E_i) < -self.epsilon and (alpha[i] < self.C)) or \
                        ((y_i * E_i) > self.epsilon and (alpha[i] > 0)):
                    j = self._select_j_rand(i, m)  # 简化版本，随机选取一个j
                    X_j, y_j = X[j, :], y[j]
                    # 计算error
                    E_j = self._E(X_j, y_j, self.w, self.b)

                    alpha_i_old, alpha_j_old = alpha[i].copy(), alpha[j].copy()
                    (L, H) = self._compute_L_H(self.C, alpha[i], alpha[j], y_i, y_j)

                    if L == H:
                        continue

                    K_ij = 2 * kernel(X_i, X_j) - kernel(X_i, X_i) - kernel(X_j, X_j)
                    if K_ij >= 0:
                        continue

                    # 更新alpha
                    alpha[j] -= float(y_j * (E_i - E_j)) / K_ij
                    alpha[j] = max(alpha[j], L)
                    alpha[j] = min(alpha[j], H)

                    if abs(alpha[j] - alpha_j_old) < 0.00001:
                        print('j not moving enough')
                        continue

                    alpha[i] += y_j * y_i * (alpha_j_old - alpha[j])

                    alphaParisChanged += 1
                    print('iter: %d i: %d, paris changed %d' % (iter, i, alphaParisChanged))

            # 检查是否已经收敛
            # diff = np.linalg.norm(alpha - alpha_prev)
            # if diff < self.epsilon:
            #     print(iter)
            #     break

            if alphaParisChanged == 0:
                iter += 1
            else:
                iter = 0
            print('iteration number: %d' % iter)

        # 计算模型参数
        self.b = self._cacl_b(X, y, self.w)
        if self.kernel_type == 'linear':
            self.w = self._cacl_w(X, y, alpha)

        # 获得支持向量
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = X[alpha_idx, :]

        return support_vectors

    # predict函数
    def predict(self, X):
        return self._h(X, self.w, self.b)

    # 随机选取一个样本
    def _select_j_rand(self, i, m):
        j = i
        while (j == i):
            j = int(random.uniform(0, m))
        return j

    def _compute_L_H(self, C, alpha_i, alpha_j, y_i, y_j):
        if y_i != y_j:
            return (max(0, alpha_j - alpha_i), min(C, C + alpha_j - alpha_i))
        else:
            return (max(0, alpha_j + alpha_i - C), min(C, alpha_j + alpha_i))

    # 计算w
    def _cacl_w(self, X, y, alpha):
        return np.dot(X.T, (alpha * y.reshape((-1, 1))))

    # 计算b
    def _cacl_b(self, X, y, w):
        b_tmp = y.reshape((-1, 1)) - np.dot(X, w)
        return np.mean(b_tmp)

    # 计算error
    def _E(self, X_k, y_k, w, b):
        return self._h(X_k, w, b) - y_k

    # predict
    def _h(self, X, w, b):
        return np.sign(np.dot(X, w) + b).astype(int)

    # kernel function
    def _kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)

    def _kernel_quadratic(self, x1, x2):
        return (np.dot(x1, x2.T) ** 2)

    def _kernel_rbf(self, x1, x2, sigma=5.0):
        if self.sigma:
            sigma = self.sigma

        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))
