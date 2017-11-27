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