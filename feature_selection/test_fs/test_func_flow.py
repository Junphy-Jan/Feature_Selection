from auto_feature_select.feature_selection import select_feature
from fs_util.func import SupportMethods, WrapTestFunc
# import pytest

best_5_feature = None


@WrapTestFunc(purpose="测试默认参数特征选择，默认选择过滤式方法")
def test_filter_methods():
    feature_score = select_feature("../data/cs-training-no-index-no-str.csv",
                                   label_col_name="SeriousDlqin2yrs", cluster_address=None)
    print("feature_score:{}".format(feature_score))


@WrapTestFunc(purpose="测试全部特征选择方法")
def test_all_methods():
    feature_score = select_feature("../data/cs-training-no-index-no-str.csv", methods=["all"],
                                   label_col_name="SeriousDlqin2yrs", cluster_address=None)
    print("feature_score:{}".format(feature_score))


@WrapTestFunc(purpose="测试选择指定的特征选择方法")
def test_special_methods():
    feature_score = select_feature("../data/cs-training-no-index-no-str.csv", methods=[SupportMethods.STEP_WISE.name],
                                   label_col_name="SeriousDlqin2yrs", cluster_address=None)
    print("feature_score:{}".format(feature_score))
    global best_5_feature
    best_5_feature = feature_score


def test_selected_feature_effect():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import numpy as np
    import random
    import time
    np.random.seed(42)
    random.seed(42)

    best_5_feature_idx = [idx[0] for idx in best_5_feature[:5]]
    data = pd.read_csv("../data/cs-training-no-index-no-str.csv")
    label_name = "SeriousDlqin2yrs"
    features = data.columns.values
    features = np.delete(features, np.where(features == label_name))

    # 选出的特征列
    used_col = features[best_5_feature_idx].tolist()
    used_col.append(label_name)
    print("选择的特征（包含标签列）：{}".format(used_col))
    not_used_col = [i for i in features if i not in used_col]
    not_used_col.append(label_name)
    print("未选择的特征（包含标签列）：{}".format(not_used_col))
    selected_data = pd.read_csv("../data/cs-training-no-index-no-str.csv",
                                usecols=used_col)

    not_selected_data = pd.read_csv("../data/cs-training-no-index-no-str.csv",
                                    usecols=not_used_col)
    data = data.fillna(0)
    selected_data = selected_data.fillna(0)
    not_selected_data = not_selected_data.fillna(0)

    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    selected_train_data, selected_test_data = train_test_split(selected_data, test_size=0.1, random_state=42)
    not_selected_train_data, not_selected_test_data = train_test_split(not_selected_data, test_size=0.1,
                                                                       random_state=42)

    start1 = time.time()
    clf1 = LogisticRegression(verbose=0, max_iter=400, C=0.5, tol=0.1).fit(train_data.drop(columns=label_name),
                                                                     train_data[label_name])
    base_score = clf1.score(test_data.drop(columns=label_name), test_data[label_name])
    cost1 = time.time() - start1

    start2 = time.time()
    clf2 = LogisticRegression(verbose=0, max_iter=400, C=0.5, tol=0.1).fit(selected_train_data.drop(columns=label_name),
                                                                     selected_train_data[label_name])
    selected_score = clf2.score(selected_test_data.drop(columns=label_name), selected_test_data[label_name])
    cost2 = time.time() - start2

    start3 = time.time()
    clf3 = LogisticRegression(verbose=0, max_iter=400, C=0.5, tol=0.1).fit(not_selected_train_data.drop(columns=label_name),
                                                                     not_selected_train_data[label_name])
    not_selected_score = clf3.score(not_selected_test_data.drop(columns=label_name), not_selected_test_data[label_name])
    cost3 = time.time() - start3

    print("全部特征,大小：{} 逻辑回归分数：{}, 耗时：{}".format(train_data.shape, base_score, cost1))

    print("特征选择后,大小：{} 逻辑回归分数：{}, 耗时：{}".format(selected_train_data.shape, selected_score, cost2))

    print("未选择的特征,大小：{} 逻辑回归分数：{}, 耗时：{}".format(not_selected_train_data.shape, not_selected_score, cost3))


if __name__ == '__main__':
    test_filter_methods()
    # test_all_methods()
    # test_special_methods()
    # test_selected_feature_effect()
