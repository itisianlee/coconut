# coding:utf8
import numpy as np


def genLinearSeparableData(weights, numLines):
    """ (list, int) -> array
    Return a linear Separable data set.(-10,10)
    Randomly generate numLines points on both sides of
    the hyperplane weights * x = 0.
    weights: include (w0,w1,...,b)
    Notice: weights and x are vectors.

    >>> data = genLinearSeparableData([2,3],5)
    >>> data
    array([[ 0.54686091,  3.60017244,  1.        ],
           [ 2.0201362 ,  7.5046425 ,  1.        ],
           [-3.14522458, -7.19333582, -1.        ],
           [ 9.72172678, -7.99611918, -1.        ],
           [ 9.68903615,  2.10184495,  1.        ]])
    """
    w = np.array(weights)
    numFeatures = len(weights)
    dataset = np.zeros((numLines, numFeatures + 1))
    for i in range(numLines):
        x = np.random.rand(1, numFeatures - 1) * 20 - 10
        x = np.append(x, 1)
        innerProduct = np.sum(w * x)
        if innerProduct <= 0:
            dataset[i] = np.append(x, -1)
        else:
            dataset[i] = np.append(x, 1)

    return dataset
