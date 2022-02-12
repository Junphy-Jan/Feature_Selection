import numpy as np
from sklearn.utils.sparsefuncs import mean_variance_axis


def cal_variances(x):
    """
    计算 特征数据 x 的方差 var = mean(abs(x - x.mean())**2)
    Args:
        x:
            {array-like, sparse matrix}, shape (n_samples, n_features)
            待计算方差的特征数据.

    Returns: 每个特征的方差值
    """
    if hasattr(x, "toarray"):  # sparse matrix
        _, variances_ = mean_variance_axis(x, axis=0)
    else:
        variances_ = np.nanvar(x, axis=0)

    if np.all(~np.isfinite(variances_)):
        if x.shape[0] == 1:
            msg = " (x:{} contains only one sample)"
            raise ValueError(msg.format(x))
    return variances_


if __name__ == '__main__':
    test_data = np.random.random((20, 3))
    test_data = np.concatenate((test_data, np.zeros((20, 1))), axis=1)
    nan_data = np.array([[0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [np.nan],
                         [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1]])
    test_data = np.concatenate((test_data, nan_data), axis=1)
    print(test_data.shape)
    print(cal_variances(test_data))

    print(test_data[:, 0].shape)
    print(cal_variances(test_data[:, 0]))
