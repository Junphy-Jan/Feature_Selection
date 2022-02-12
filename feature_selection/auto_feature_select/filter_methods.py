import math
import os
import numpy as np
import random
from typing import Union, List
import enum
import ray
import time
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
import copy

from auto_feature_select._base import FeatureSelectionActor
from filter_selection import chi2, mutual_info_classif, mutual_info_regression
from filter_selection.cal_variances import cal_variances
from filter_selection.iv.information_value import cal_iv
from fs_util.func import SupportMethods, is_continuous


@ray.remote
def task_iv(label_data, feature_data, feature_name):
    """
    计算 特征数据与标签数据的 IV值
    Args:
        label_data: 标签数据 {array-like}, shape (n_samples,)
        feature_data: 特征数据 {array-like}, shape (n_samples,)
        feature_name: 特征名

    Returns:
        方法名，IV值
    """
    print("进程：{}使用{}方法对特征：{}进行处理！".format(os.getpid(), SupportMethods.IV.name, feature_name))
    iv = cal_iv(feature_data, label_data, n_bins=10)
    return SupportMethods.IV.name, iv


# @ray.remote
def task_null_rate(feature_data: np.ndarray, feature_name, nan_replace):
    print("进程：{}使用{}方法对特征：{}进行处理！".format(os.getpid(), SupportMethods.NULL_FILTER.name, feature_name))
    feature_size = feature_data.shape[0]
    if feature_size == 0:
        return SupportMethods.NULL_FILTER.name, 1
    null_position = []
    feature_data_copy = None
    for i in range(feature_size):
        try:
            if np.isnan(float(feature_data[i])) or np.isinf(float(feature_data[i])):
                if feature_data_copy is None:
                    feature_data_copy = feature_data.copy()
                    feature_data_copy[i] = 0
                else:
                    feature_data_copy[i] = 0
                null_position.append(i)
        except ValueError as e:
            # 非 数值型数据
            if feature_data_copy is None:
                feature_data_copy = feature_data.copy()
                feature_data_copy[i] = 0
            else:
                feature_data_copy[i] = 0
            null_position.append(i)

    feature_null_count = len(null_position)
    null_rate = feature_null_count / feature_size

    # 填充缺失值
    if feature_null_count > 0:
        feature_data_copy = feature_data_copy.astype(np.float32)
        # feature_data_copy = copy.deepcopy(feature_data)
        if isinstance(nan_replace, str):
            if nan_replace == "avg":
                start = time.time()
                try:
                    # 平均值取整
                    nan_replace = int(np.sum(feature_data_copy) / feature_size)
                except TypeError:
                    print(feature_data_copy)
        print("缺失值nan、inf 等将在 过滤阶段 被替换为：{}".format(nan_replace))
        feature_data_copy[null_position] = nan_replace
    else:
        feature_data_copy = feature_data
    return SupportMethods.NULL_FILTER.name, null_rate, feature_data_copy


@ray.remote
def task_variance(feature_data, feature_name):
    print("进程：{}使用{}方法对特征：{}进行处理！".format(os.getpid(), SupportMethods.VARIANCE.name, feature_name))
    return SupportMethods.VARIANCE.name, cal_variances(feature_data)


@ray.remote
def task_pearson_corr(label_data, feature_data, feature_name):
    """
    Args:
        label_data: (N,) array_like 标签数据
        feature_data: (N,) array_like 某特征所有数据
        feature_name: 特征名

    Returns:
        (方法名, 皮尔逊系数, p-value).
    """
    print("进程：{}使用{}方法对特征：{}进行处理！".format(os.getpid(), SupportMethods.PEARSON_CORRELATION.name, feature_name))
    pearson_corr, p_value = pearsonr(feature_data, label_data)
    if np.isnan(pearson_corr) and np.isnan(p_value):
        pearson_corr, p_value = 0, 0
    # print("特征：{} 的 Pearson 系数：{}".format(feature_name, pearson_corr))
    return SupportMethods.PEARSON_CORRELATION.name, math.fabs(pearson_corr), p_value


@ray.remote
def task_chi2(label_data, feature_data, feature_name):
    """
    Args:
        label_data: (N,) array_like 标签数据
        feature_data: (N,) array_like 某特征所有数据
        feature_name: 特征名

    Returns:
        (方法名，卡方值，p-value)
    """
    print("进程：{}使用{}方法对特征：{}进行处理！".format(os.getpid(), SupportMethods.CHI2.name, feature_name))
    try:
        chi2_value, p_value = chi2(feature_data.reshape(-1, 1), label_data)
        if np.isnan(chi2_value[0]):
            chi2_value[0] = 0
        return SupportMethods.CHI2.name, chi2_value[0]
    except ValueError as e:
        # print("计算卡方值出现异常：{}，将跳过".format(e))
        if e.args[0] == "Input X must be non-negative.":
            return SupportMethods.CHI2.name, 0


@ray.remote
def task_mutual_info(label_data, feature_data, feature_name, n_neighbors=3, x_discrete: Union[str, bool] = "auto",
                     y_discrete: Union[str, bool] = "auto", copy=True, random_state=None):
    """
        Args:
            label_data: (N,) array_like 标签数据
            feature_data: (N,) array_like 某特征所有数据
            feature_name: 特征名
            n_neighbors: 最近邻，默认为 3
            y_discrete: y 标签是否是离散型变量，如分类则设置为 True
            x_discrete: 特征数据是否是离散型变量。
            copy: 是否拷贝数据
            random_state: 随机种子
        Returns:
            (方法名，卡方值，p-value)
        """
    print("进程：{}使用{}方法对特征：{}进行处理！".format(os.getpid(), SupportMethods.MUTUAL_INFO.name, feature_name))
    if y_discrete:
        mi = mutual_info_classif(feature_data.reshape(-1, 1), label_data, discrete_features=x_discrete,
                                 n_neighbors=n_neighbors, copy=copy, random_state=random_state)
    else:
        mi = mutual_info_regression(feature_data.reshape(-1, 1), label_data, discrete_features=x_discrete,
                                    n_neighbors=n_neighbors, copy=copy, random_state=random_state)
    # print("特征：{} 的互信息：{}".format(feature_name, mi))
    return SupportMethods.MUTUAL_INFO.name, mi[0]


@ray.remote
class FilterActor(FeatureSelectionActor):
    def __init__(self, actor_name: str, methods_name: List[str], *, select_best_n: int = None,
                 keep_feature: Union[str, List[str]] = None):
        super().__init__(actor_name, select_best_n, keep_feature)
        # 记录结果：{"特征0"：{"方法1":结果, "方法2":结果,...}, "特征1"：{"方法1":结果, "方法2":结果,...}, ...}
        self.methods_name = methods_name
        self.filter_recorder = {}
        self.skipped_feature = []

    def work(self, *, label_data, feature_data, feature_name, nan_replace, null_rate_limit,
             n_neighbors=3, x_discrete="auto", y_discrete="auto"):
        # print("enter filter worker's work method")
        ret_data = []
        null_rate = task_null_rate(feature_data, feature_name, nan_replace)
        # null_rate = ray.get(task_null_rate_f)
        if null_rate[1] > null_rate_limit:
            self.skipped_feature.append(feature_name)
            print("特征：{} 的数据空值率超过：{}，将跳过计算该列！".format(feature_name, null_rate_limit))
            return
        else:
            ret_data.append(null_rate)
        feature_data = null_rate[2]
        methods = []
        if self.methods_name.__contains__(SupportMethods.MUTUAL_INFO.name):
            methods.append(task_mutual_info.remote(label_data, feature_data, feature_name, n_neighbors=n_neighbors,
                                                   x_discrete=x_discrete, y_discrete=y_discrete))
        if self.methods_name.__contains__(SupportMethods.IV.name):
            methods.append(task_iv.remote(label_data, feature_data, feature_name))
        if self.methods_name.__contains__(SupportMethods.VARIANCE.name):
            methods.append(task_variance.remote(feature_data, feature_name))
        if self.methods_name.__contains__(SupportMethods.PEARSON_CORRELATION.name):
            methods.append(task_pearson_corr.remote(label_data, feature_data, feature_name))
        if self.methods_name.__contains__(SupportMethods.CHI2.name):
            methods.append(task_chi2.remote(label_data, feature_data, feature_name))
        if len(methods) > 0:
            ret_data.extend(ray.get(methods))
        print("进程：{}使用 {} 方法处理完成，返回数据：{}！".format(os.getpid(), self.actor_name, ret_data))
        ret_dic = {}
        for ret in ret_data:
            ret_dic[ret[0]] = ret[1]
        self.filter_recorder[feature_name] = ret_dic

    def get_attr(self):
        return self.filter_recorder


if __name__ == '__main__':
    X_train = np.load("D:\\data\\ray_lr\\mnist_dataset_csv\\mnist_test_x.npy")
    y_train = np.load("D:\\data\\ray_lr\\mnist_dataset_csv\\mnist_test_y.npy")
    np.random.seed(1)
    random.seed(1)
    from sklearn.datasets import load_breast_cancer
    import pandas as pd

    # data = load_breast_cancer()
    """data = pd.DataFrame(np.c_[data['data'], data['target']],
                        columns=np.append(data['feature_names'], ['target']))
    X_train, X_test, y_train, y_test = train_test_split(data.drop(labels=['target'], axis=1),
                                                        data.target, test_size=0.2,
                                                        random_state=0)"""
    my_mutual_info = []
    for i_ in range(X_train.shape[1]):
        # my_mutual_info.append(task_mutual_info(y_train.values, X_train.iloc[:, i].values, "feature-" + str(i))[1])
        my_mutual_info.append(
            task_mutual_info(y_train, X_train[:, i_], "feature-" + str(i_), x_discrete=True, y_discrete=True)[1])

    print("my mutual info :{}".format(my_mutual_info))
    discrete_features = [True] * X_train.shape[1]
    sk_mutual_info = mutual_info_classif(X_train, y_train, discrete_features=np.array(discrete_features))
    mutual_info_regression()
    print("sk_mutual_info :{}".format(sk_mutual_info))
    print("my mutual info top 20:{}".format(np.array(my_mutual_info).argsort()[-10:][::-1]))
    print("sk_mutual_info top 20:{}".format(sk_mutual_info.argsort()[-10:][::-1]))
