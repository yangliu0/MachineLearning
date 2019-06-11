# -*- coding:utf-8 -*-
# @Time 2019/5/24
# @Author liuyang

# kmeans算法的简单实现
import numpy as np


# 读取文件，转换为list
def load_dataset(file_name):
    data_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        float_line = list(map(float, cur_line))
        data_mat.append(float_line)
    return data_mat


# 欧式距离
def dist_calc(vec_a, vec_b):
    return np.sqrt(np.sum(np.power(vec_a - vec_b, 2)))


# 随机生成k个质心
def rand_cent(dataset, k):
    n = dataset.shape[1]
    # 质心
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        min_j = np.min(dataset[:, j])
        range_j = np.float(np.max(dataset[:, j]) - min_j)
        # 控制范围，使得每一维度生成的随机数不超过数据集中这一维度的范围
        centroids[:, j] = np.mat(min_j + range_j * np.random.rand(k, 1))
    return centroids


def kmeans(dataset, k, dist_measure=dist_calc, create_cent=rand_cent):
    """
    :param dataset: 加载的数据，np.mat类型
    :param k: 簇的个数
    :param dist_measure: 距离度量方法
    :param create_cent: 初始化质心方法
    :return:
    """
    m = dataset.shape[0]
    # 保存每一个数据点被分配给的簇心及距离
    cluster_assment = np.mat(np.zeros((m, 2)))

    centroids = create_cent(dataset, k)
    # 标记簇心是否变化
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            # 对每个数据点，将其分配给最近的簇心
            min_dist = np.inf
            min_index = -1
            for j in range(k):
                dist_ji = dist_measure(centroids[j, :], dataset[i, :])
                if dist_ji < min_dist:
                    min_dist = dist_ji
                    min_index = j
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
            cluster_assment[i, :] = [min_index, min_dist ** 2]
        print(centroids)
        for cent in range(k):
            # 获取到归属于本簇的所有数据点
            point_in_cluster = dataset[np.nonzero(cluster_assment[:, 0] == cent)[0]]
            # 将簇心修改为属于本簇的点的平均值
            centroids[cent, :] = np.mean(point_in_cluster, axis=0)
    return centroids, cluster_assment


def bi_kmeans(dataset, k, dist_measure=dist_calc):
    """
    :param dataset: 输入数据，数据类型为np.mat
    :param k: 簇的个数
    :param dist_measure: 距离度量方法
    :return:
    """
    m = dataset.shape[0]
    # 保存每一个数据点的分类结果和平方误差
    cluster_assment = np.mat(np.zeros((m, 2)))
    centroid0 = np.mean(dataset, axis=0).tolist()[0]
    cent_list = [centroid0]
    for j in range(m):
        cluster_assment[j, 1] = dist_measure(np.mat(centroid0), dataset[j, :]) ** 2
    while len(cent_list) < k:
        # 最小误差平方和
        lowest_sse = np.inf
        for i in range(len(cent_list)):
            point_in_curr_cluster = dataset[np.nonzero(cluster_assment[:, 0] == i)[0], :]
            # 尝试划分每一簇，保存SSE最小的划分
            centroid_mat, split_clust_ass = kmeans(point_in_curr_cluster, 2, dist_measure)
            sse_split = np.sum(split_clust_ass[:, 1])
            sse_not_split = np.sum(cluster_assment[np.nonzero(cluster_assment[:, 0] != i)[0], 1])
            print('sse_split and sse_not_split {} {}'.format(sse_split, sse_not_split))
            if (sse_split + sse_not_split) < lowest_sse:
                best_cent_to_split = i
                best_new_cents = centroid_mat
                best_clust_ass = split_clust_ass.copy()
                lowest_sse = sse_split + sse_not_split
        # 更新簇的分配结果
        best_clust_ass[np.nonzero(best_clust_ass[:, 0] == 1)[0], 0] = len(cent_list)
        best_clust_ass[np.nonzero(best_clust_ass[:, 0] == 0)[0], 0] = best_cent_to_split
        print('best_cent_to_split {}'.format(best_cent_to_split))
        print('length of best_clust_ass {}'.format(len(best_clust_ass)))
        cent_list[best_cent_to_split] = best_new_cents[0, :].tolist()[0]
        cent_list.append(best_new_cents[1, :].tolist()[0])
        cluster_assment[np.nonzero(cluster_assment[:, 0] == best_cent_to_split)[0], :] = best_clust_ass
    return np.mat(cent_list), cluster_assment


def main():
    data_mat1 = np.mat(load_dataset('testSet.txt'))
    data_mat2 = np.mat(load_dataset('testSet2.txt'))
    # centroids, cluster_assment = kmeans(data_mat, 4)

    cent_list, cluster_assment = bi_kmeans(data_mat2, 3)


if __name__ == '__main__':
    main()
