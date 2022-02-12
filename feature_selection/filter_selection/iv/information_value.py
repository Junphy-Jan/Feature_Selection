import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree

from fs_util.func import is_continuous

DEFAULT_BINS = 6


def bin_by_splits(feature, splits, nan):
    """Bin feature by split points
    """
    feature = np.nan_to_num(feature, nan=nan, copy=True)
    if not isinstance(splits, (list, np.ndarray)):
        splits = [splits]

    return np.digitize(feature, splits)

def dt_bins(feature: np.ndarray, target,
            nan=-1, n_bins=None, min_samples=1, **kwargs):
    """ 使用决策树分箱
    Args:
        feature (array-like)
        target (array-like): target will be used to fit decision tree
        nan (number): value will be used to fill nan
        n_bins (int): n groups that will be merged into
        min_samples (int): min number of samples in each leaf nodes
    Returns:
        array: array of split points
    """
    if n_bins is None and min_samples == 1:
        n_bins = DEFAULT_BINS

    # feature = fillna(feature, by=nan)
    feature = np.nan_to_num(feature, copy=True, nan=nan)

    tree = DecisionTreeClassifier(
        min_samples_leaf=min_samples,
        max_leaf_nodes=n_bins,
    )

    tree.fit(feature.reshape((-1, 1)), target)

    thresholds = tree.tree_.threshold
    thresholds = thresholds[thresholds != _tree.TREE_UNDEFINED]
    splits = np.sort(thresholds)
    if len(splits):
        bins = bin_by_splits(feature, splits, nan)
    else:
        bins = np.zeros(len(feature))
    return bins

def response_count(arr, value, smoothing_term=None):
    c = (arr == value).sum()
    if smoothing_term is not None and c == 0:
        return smoothing_term
    return c


def quantity(target, mask=None):
    """get probability of target by mask
    """
    if mask is None:
        return 1, 1
    sub_target = target[mask]
    good_i = response_count(sub_target, 0, smoothing_term=1)
    bad_i = response_count(sub_target, 1, smoothing_term=1)
    return bad_i, good_i

def WOE(bad_prob, good_prob):
    """get WOE of a group
    Args:
        bad_prob: the probability of grouped bad in total bad
        good_prob: the probability of grouped good in total good
    Returns:
        number: woe value
    """
    return np.log(bad_prob / good_prob)

def _IV(feature, target):
    """计算IV值  "1|bad"

    Args:
        feature np.ndarray
        target np.ndarray

    Returns:
        feature_iv: IV
        iv: IV of each groups
    """
    iv = {}
    good_total = response_count(target, 0, smoothing_term=1)
    bad_total = response_count(target, 1, smoothing_term=1)
    for v in np.unique(feature):
        bad_i, good_i = quantity(target, mask=(feature == v))
        bad_prob = bad_i / bad_total
        good_prob = good_i / good_total
        iv[v] = (bad_prob - good_prob) * WOE(bad_prob, good_prob)

    feature_iv = sum(list(iv.values()))
    return feature_iv, iv


def cal_iv(feature, target, nan=-1, n_bins=None, **kwargs):
    """计算 feature 与 target 的IV值

    Args:
        feature: 1d ndarray 特征数据
        target: 1d ndarray 标签数据
        nan: 对 空值的填补值
        n_bins: 决策树分箱数，默认为6

    Returns:
        iv : float 特征数据与标签数据的IV值.
    """
    if is_continuous(feature):
        feature = dt_bins(feature, target, nan=nan, n_bins=n_bins, **kwargs)

    iv, group_iv = _IV(feature, target)
    # print("feature:{} 总IV值：{}， 各组IV值：{}".format(kwargs["feature_name"], iv, group_iv))
    return iv
