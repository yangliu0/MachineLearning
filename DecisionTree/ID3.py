# -*- coding:utf-8 -*-

import numpy as np


# ID3 Decision Tree

class DecisionTree:
    def __init__(self):
        self._tree = None

    def _calcEntropy(self, y):
        """
        计算香农熵
        :param y: 数据集标签
        :return: 香农熵
        """

        num = y.shape[0]
        # 统计y中不同label值的个数，并用字典labelCounts存储
        labelCounts = {}
        for label in y:
            if label not in labelCounts.keys():
                labelCounts[label] = 0
            labelCounts[label] += 1

        # 计算熵
        entropy = 0.0
        for key in labelCounts:
            prop = float(labelCounts[key]) / num
            entropy -= prop * np.log2(prop)
        return entropy

    def _splitDataset(self, X, y, index, value):
        """
        :param X:
        :param y:
        :param index:
        :param value:
        :return: 数据集中特征下标为index，特征值为value的子数据集
        """

        ret = []
        featVec = X[:, index]
        X = X[:, [i for i in range(X.shape[1]) if i != index]]
        for i in range(len(featVec)):
            if featVec[i] == value:
                ret.append(i)
        return X[ret, :], y[ret]

    def _chooseBestFeatureToSplit(self, X, y):
        """
        :param X:
        :param y:
        :return: bestFeature的index
        """
        numFeatures = X.shape[1]
        oldEntropy = self._calcEntropy(y)
        bestInfoGain = 0.0
        bestFeatureIndex = -1
        # 对每个feature都计算一下信息增益，并用bestInfoGain记录最大的那个
        for i in range(numFeatures):
            featureList = X[:, i]
            uniqueVals = set(featureList)
            newEntropy = 0.0
            # 计算每种划分方式的信息熵
            for value in uniqueVals:
                sub_X, sub_y = self._splitDataset(X, y, i, value)
                prob = len(sub_y) / float(len(y))
                newEntropy += prob * self._calcEntropy(sub_y)
            # 计算信息增益
            infoGain = oldEntropy - newEntropy
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeatureIndex = i
        return bestFeatureIndex

    def _majorityCnt(self, labelList):
        """
        :param labelList:
        :return: labelList中出现次数最多的label
        """

        labelCount = {}
        for vote in labelList:
            if vote not in labelCount.keys():
                labelCount[vote] = 0
            labelCount[vote] += 1
        sortedLabelCount = sorted(labelCount.items(), key=lambda x: x[1], reverse=True)
        return sortedLabelCount[0][0]

    def _createTree(self, X, y, featureIndex):
        """
        构建决策树
        :param X:
        :param y:
        :param featureIndex: 元组，记录X中特征在原始数据中对应index
        :return:
        """
        labelList = list(y)
        # 所有类别都相同，返回该label
        if labelList.count(labelList[0]) == len(labelList):
            return labelList[0]
        # 没有特征可分割，返回出现次数最多的label
        if len(featureIndex) == 0:
            return self._majorityCnt(labelList)

        # 可以继续分割，找寻最佳分割点
        bestFeatureIndex = self._chooseBestFeatureToSplit(X, y)
        bestFeatStr = featureIndex[bestFeatureIndex]
        featureIndex = list(featureIndex)
        featureIndex.remove(bestFeatStr)
        featureIndex = tuple(featureIndex)

        # 用字典存储决策树
        myTree = {bestFeatStr: {}}
        featValues = X[:, bestFeatureIndex]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            # 对每个value递归建树
            sub_X, sub_y = self._splitDataset(X, y, bestFeatureIndex, value)
            myTree[bestFeatStr][value] = self._createTree(sub_X, sub_y, featureIndex)
        return myTree

    def fit(self, X, y):
        # 类型检查
        if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            pass
        else:
            try:
                X = np.array(X)
                y = np.array(y)
            except:
                raise TypeError("numpy.ndarray requied for X, y")

        featureIndex = tuple(['x' + str(i) for i in range(X.shape[1])])
        self._tree = self._createTree(X, y, featureIndex)

    def predict(self, X):
        if self._tree == 'None':
            raise NotImplementedError("Estimator not fitted, call 'fit' first")

        # 类型检查
        if isinstance(X, np.ndarray):
            pass
        else:
            try:
                X = np.array(X)
            except:
                raise TypeError("numpy.ndarray requied for X")

        def _classify(tree, sample):
            """
            用训练好的决策树对输入数据分类
            :param tree:
            :param sample:
            :return:
            """
            featureIndex = list(tree.keys())[0]
            secondDict = tree[featureIndex]
            key = sample[int(featureIndex[1:])]
            valueOfKey = secondDict[key]
            if isinstance(valueOfKey, dict):
                label = _classify(valueOfKey, sample)
            else:
                label = valueOfKey
            return label

        if X.shape[0] == 1:
            return _classify(self._tree, X)
        else:
            result = []
            for i in range(X.shape[0]):
                result.append(_classify(self._tree, X[i]))
            return np.array(result)

    def show(self):
        if self._tree == None:
            raise NotImplementedError("Estimator not fitted, call 'fit' first")

        # plot the tree using matplotlib
        import treePlotter
        print(self._tree)
        treePlotter.createPlot(self._tree)


if __name__ == '__main__':
    clf = DecisionTree()
    X = [[1, 2, 0, 1, 0],
         [0, 1, 1, 0, 1],
         [1, 0, 0, 0, 1],
         [2, 1, 1, 0, 1],
         [1, 1, 0, 1, 1]]
    y = ['yes', 'yes', 'no', 'no', 'no']
    clf.fit(X, y)
    clf.predict(X)
    clf.show()
