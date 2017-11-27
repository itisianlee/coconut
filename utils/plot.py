# coding:utf8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plotData(dataSet, w=None):
    """
    画出数据集的点,以及预测的超平面
    :param dataSet: 要画出的点
    :param w: 权重
    :return: 无
    """
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.set_title('Linear separable data set')
    plt.xlabel('X')
    plt.ylabel('Y')
    labels = np.array(dataSet[:, 2])
    idx_1 = np.where(dataSet[:, -1] == 1)
    p1 = ax.scatter(dataSet[idx_1, 0], dataSet[idx_1, 1], marker='o', color='g', label=1, s=20)
    idx_2 = np.where(dataSet[:, -1] == -1)
    p2 = ax.scatter(dataSet[idx_2, 0], dataSet[idx_2, 1], marker='x', color='r', label=-1, s=20)

    if w is not None:
        x_ = -1 * (w[-1] + w[1] * 12) / w[0]
        coord1 = (x_, 10)
        x_ = -1 * (w[-1] - w[1] * 12) / w[0]
        coord2 = (x_, -10)
        line1 = [coord1, coord2]
        (line1_xs, line1_ys) = zip(*line1)
        ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='blue', label='pred'))
    plt.legend(loc='upper right')
    plt.show()
