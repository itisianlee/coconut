# coding:utf8
import numpy as np


def pla(dataset, dim=2, lr=1):
    """
    感知机算法原始形式实现
    :param dataset:
    :param dim:
    :param lr:
    :return: 得到的权重
    """
    w = np.zeros(dim + 1)
    # W = np.random.rand(dim) * 10
    flag = True
    while flag:
        flag = False
        for d in dataset:
            if d[-1] * np.sum((w * d[:-1])) <= 0:
                w = w + lr * d[-1] * d[:-1]
                flag = True
                break
    return w


def plaDualForm(dataset, lr=1.):
    """
    感知机算法对偶形式实现
    :param dataset:
    :param dim:
    :param lr:
    :return: 权重
    """
    N = dataset.shape[0]  # 数据规模
    x = dataset[:, :-2]
    G = np.dot(x, x.T)  # 计算Gram矩阵
    alpha = np.zeros(N)
    b = 0.
    flag = True
    while flag:
        flag = False
        for i in range(N):
            d = dataset[i]
            if d[-1] * (np.sum((alpha * dataset[:, -1] * G[i])) + b) <= 0:
                alpha[i] = alpha[i] + lr
                b = b + lr * d[-1]
                flag = True
                break
    w = np.sum(np.multiply(np.reshape(alpha * dataset[:, -1], (N, 1)), x), axis=0)
    return np.append(w, b)
