import random

import ray
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import toad

from filter_selection.iv.information_value import cal_iv
from fs_util.func import WrapTestFunc

data = pd.read_csv("D:\\data\\kaggle\\giveMeSomeCredit\\cs-training-no-index.csv")
# ray.init()
# data = pd.read_csv("D:\\data\\ray_lr\\mnist_dataset_csv\\mnist_train_norm_binary.csv")
# data = data[:30000]
print(data.shape)

def test_filtered_classify(x, y, dropped_col_name):
    np.random.seed(1)
    random.seed(1)
    clf = LogisticRegression(verbose=0, max_iter=150, tol=1e-5).fit(x, y)
    test_data = pd.read_csv("D:\\data\\ray_lr\\mnist_dataset_csv\\mnist_test_norm_binary.csv")
    label_data = test_data["label"].values
    cols = test_data.columns.copy().drop(dropped_col_name)
    test_data = test_data[cols].values
    return clf.score(test_data, label_data)

@WrapTestFunc(purpose="测试 toad 库计算IV值")
def test_toad_lib_iv():
    _, dropped_features, dropped_feature_iv_values = \
        toad.selection.drop_iv(data, "SeriousDlqin2yrs", return_drop=True, return_iv=True, threshold=0.015)
    print("toad库分箱 丢失特征数：{}, IV值：{}"
          .format(len(dropped_features), dropped_feature_iv_values))

@ray.remote
def remote_cal_iv(feature_data, label, col_name):
    kwargs = {"feature_name": col_name}
    return cal_iv(feature_data, label, **kwargs)

@WrapTestFunc(purpose="测试 优化后的计算IV值方法")
def test_self_iv():
    count = 0
    dropped_features = []
    for feature in data.keys():
        if feature == "label":
            continue
        feature_name = {"feature_name": feature}
        iv = cal_iv(data[feature].values, data["label"].values, n_bins=20, **feature_name)
        if iv < 0.005:
            count += 1
            dropped_features.append(feature)
        # print("feature:{}\t\t总IV值：{}".format(feature_name, iv))
    print("my iv calculator 丢弃数：{}".format(count))
    dropped_features.append("label")
    filtered_cols = data.columns.copy().drop(dropped_features)
    after_filter_x = data[filtered_cols].values
    after_filter_y = data["label"].values
    print("toad 库丢弃特征后测试，train_x.shape:{}, 结果：{}"
          .format(after_filter_x.shape, test_filtered_classify(after_filter_x, after_filter_y, dropped_features)))

@WrapTestFunc(purpose="测试特征丢弃前的模型结果")
def test_original_data_performance():
    filtered_cols = data.columns.copy().drop("label")
    before_filter_x = data[filtered_cols].values
    before_filter_y = data["label"].values
    print("丢弃特征前测试，train_x.shape:{}, 结果：{}"
          .format(before_filter_x.shape, test_filtered_classify(before_filter_x, before_filter_y, "label")))


if __name__ == '__main__':
    # test_original_data_performance()
    test_toad_lib_iv()
    # test_self_iv()


    """columns = list(data.keys())
    start = time.time()
    remote_task = []
    count = 0
    for i in range(1, len(columns)):
        remote_task.append(remote_cal_iv.remote(data[columns[i]].values, data["label"].values, columns[i]))
        if i % 80 == 0 or i == len(columns) - 1:
            remote_task_ret = ray.get(remote_task)
            for j in remote_task_ret:
                if j < 0.02:
                    count += 1
            remote_task = []
        # print("feature:{}\t\t总IV值：{}".format(feature_name, iv))
    print("my remote iv calculator cost:{}，丢弃数：{}".format(time.time() - start, count))"""

    """filter_count = 0
    dropped_features = ["SeriousDlqin2yrs"]
    remote_ref = []
    col_length = data.keys().__len__()
    columns = list(data.keys())
    start = time.time()
    for i in range(1, col_length):
        # print(optimal_binning_boundary(x=data[data.keys()[i]], y=data['SeriousDlqin2yrs']))
        column_name = columns[i]
        remote_ref.append(feature_woe_iv.remote(x=data[column_name], y=data['SeriousDlqin2yrs'], column_name=column_name))
        if i % 4 == 0 or i == col_length-1:
            remote_ret = ray.get(remote_ref)
            # remote_ret = remote_ref
            for value, col_name in remote_ret:
                # print("特征:{} IV值：{}".format(data.keys()[i], iv_value))
                if value < 0.02:
                    filter_count += 1
                    dropped_features.append(col_name)
            remote_ref = []
    print("决策树分箱计算IV值用时：{}, 丢弃特征数：{}".format(time.time() - start, filter_count))
    filtered_cols = data.columns.copy().drop(dropped_features)
    after_filter_x = data[filtered_cols].values
    after_filter_y = data["SeriousDlqin2yrs"].values"""
    # print("决策树丢弃特征后测试，train_x.shape:{}, 结果：{}"
    #      .format(after_filter_x.shape, test_filtered_classify(after_filter_x, after_filter_y, dropped_features)))
