# coding=utf-8
import math
import random

import pytest
import shap
import toad
import pandas as pd
import numpy as np
import ray
from lightgbm import LGBMClassifier, Booster
from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold, chi2, RFE
# from sklearn.inspection import permutation_importance
from sklearn.inspection import permutation_importance
from xgboost_ray import RayDMatrix
from scorecardpy import iv
from auto_feature_select import filter_methods
from auto_feature_select.embedded_methods import lgb_ray, task_permutation_importance
from auto_feature_select.wrapper_methods import recursive_feature_elimination
from filter_selection import mutual_info_classif
from filter_selection.inspection.fs_permutation_importance import permutation_importance as fs_pi
from parse_data.read_api import FSRayDataSets

from fs_util.func import WrapTestFunc, SupportMethods
from warnings import filterwarnings
filterwarnings('ignore')

ray.shutdown()
ray.init()
data = pd.read_csv("../data/final_test.csv")
# data = pd.read_csv("D:\\data\\classify\\final_train.csv")
label_name = "Response"
# label_name = "SeriousDlqin2yrs"

dataset = FSRayDataSets(label_name)
# dataset.read_data(["D:\\data\\classify\\final_train-part1.csv", "D:\\data\\classify\\final_train-part2.csv"], transformer="modin")
dataset.read_data("../data/final_test.csv", transformer="modin")
ray_data = dataset.ds

feature_name_list = data.columns.values.tolist()
feature_name_list.remove(label_name)
np.random.seed(42)
random.seed(42)

# 传统特征排序
sk_feature_rank = {}
sk_feature_value = {}
# fs 特征排序
fs_feature_rank = {}
fs_feature_value = {}

lgb_model = None
lgb_on_ray_model = None

mock_data = np.random.randn(100000)
null_num = np.random.randint(100000, size=(1,))[0]
null_index = np.random.randint(100000, size=null_num).tolist()
null_num = len(set(null_index))
nan_list = [float('nan'), np.nan, np.inf, -np.inf]
mock_data[null_index] = random.choice(nan_list)


@WrapTestFunc(purpose="构造数据测试空值率是否准确", expect="空值率与构造数据的空值率相同")
def test_null_rate():
    task_null = filter_methods.task_null_rate.remote(mock_data, "mock data", "avg")
    _, null_rate, feature_data_copy = ray.get(task_null)
    print("数据空值率:{}， 计算得到空值率：{}".format(null_num / len(mock_data), null_rate))
    assert null_rate == null_num / len(mock_data)

@WrapTestFunc(purpose="测试scorecard库计算IV值及计算时间")
def test_scorecard_iv():

    iv_list = iv(data, y=label_name, order=False)
    print("scorecard库 IV值：\n{}"
          .format(iv_list))
    sk_feature_rank[SupportMethods.IV.name] = np.argsort(iv_list["info_value"].values)[::-1]
    sk_feature_value[SupportMethods.IV.name] = iv_list["info_value"].values


@WrapTestFunc(purpose="测试toad库计算IV值及计算时间")
def test_toad_iv(threshold=0.05):
    _, dropped_features, dropped_feature_iv_values = \
        toad.selection.drop_iv(data, label_name, return_drop=True, return_iv=True, threshold=threshold)
    print("toad库分箱 丢失特征数：{}, IV值：\n{}"
          .format(len(dropped_features), dropped_feature_iv_values))
    sk_feature_rank[SupportMethods.IV.name] = np.argsort(dropped_feature_iv_values.values)[::-1]
    sk_feature_value[SupportMethods.IV.name] = dropped_feature_iv_values.values


@WrapTestFunc(purpose="测试 fs 库计算IV值及计算时间")
def test_fs_iv():
    batch_size = 10
    epoch = len(feature_name_list) / batch_size if len(feature_name_list) % batch_size == 0 else \
        int(len(feature_name_list) / batch_size) + 1
    ivs = []
    for i in range(epoch):
        tasks = []
        for feature_name in feature_name_list[i * batch_size: (i+1) * batch_size]:
            tasks.append(
                filter_methods.task_iv.remote(ray_data[label_name], ray_data[feature_name].astype(dtype=np.float32),
                                              feature_name))
        for ret in ray.get(tasks):
            ivs.append(ret[1])
    print(ivs)
    fs_feature_rank[SupportMethods.IV.name] = np.argsort(np.array(ivs))[::-1]
    fs_feature_value[SupportMethods.IV.name] = np.array(ivs)


@WrapTestFunc(purpose="测试 sklearn 库方差计算及计算时间")
def test_sk_variance(threshold=0.01):
    sel = VarianceThreshold(threshold=threshold)
    filtered_f = sel.fit_transform(data.drop(columns=label_name))
    print("sklearn 计算的特征方差：\n{}".format(sel.variances_))
    sk_feature_rank[SupportMethods.VARIANCE.name] = np.argsort(sel.variances_)[::-1]
    sk_feature_value[SupportMethods.VARIANCE.name] = sel.variances_


@WrapTestFunc(purpose="测试 fs 库方差计算及计算时间")
def test_fs_variance():
    tasks = []
    for feature_name in feature_name_list:
        tasks.append(
            filter_methods.task_variance.remote(ray_data[feature_name].astype(dtype=np.float32), feature_name))
    variance = [ret[1] for ret in ray.get(tasks)]
    print(variance)
    fs_feature_rank[SupportMethods.VARIANCE.name] = np.argsort(np.array(variance))[::-1]
    fs_feature_value[SupportMethods.VARIANCE.name] = np.array(variance)


@WrapTestFunc(purpose="测试 scipy 库皮尔逊系数计算及计算时间")
def test_scipy_pearson():
    co_list = []
    for feature_name in feature_name_list:
        co, p_value = pearsonr(data[feature_name].astype(dtype=np.float32), data[label_name])
        co_list.append(math.fabs(co))
    print("皮尔逊系数：{}".format(co_list))
    sk_feature_rank[SupportMethods.PEARSON_CORRELATION.name] = np.argsort(np.array(co_list))[::-1]
    sk_feature_value[SupportMethods.PEARSON_CORRELATION.name] = np.array(co_list)


@WrapTestFunc(purpose="测试 fs 库皮尔逊系数计算及计算时间")
def test_fs_pearson():
    tasks = []
    for feature_name in feature_name_list:
        tasks.append(
            filter_methods.task_pearson_corr.remote(ray_data[label_name],
                                                    ray_data[feature_name].astype(dtype=np.float32),
                                                    feature_name))
    pearson_list = [ret[1] for ret in ray.get(tasks)]
    print("皮尔逊系数：{}".format(pearson_list))
    fs_feature_rank[SupportMethods.PEARSON_CORRELATION.name] = np.argsort(np.array(pearson_list))[::-1]
    fs_feature_value[SupportMethods.PEARSON_CORRELATION.name] = np.array(pearson_list)


@WrapTestFunc(purpose="测试 sk 库卡方计算及计算时间")
def test_sk_chi2():
    chi_values = []
    for feature_name in feature_name_list:
        try:
            ret = chi2(data[feature_name].values.reshape(-1, 1), data[label_name])
            chi_values.append(ret[0][0])
            # print("sk 卡方值：{}".format(ret))
        except ValueError as e:
            print(e)
            print("值为负数，跳过")
            chi_values.append(-1)
    print("sk 卡方值：{}".format(chi_values))
    sk_feature_rank[SupportMethods.CHI2.name] = np.argsort(np.array(chi_values))[::-1]
    sk_feature_value[SupportMethods.CHI2.name] = np.array(chi_values)


@WrapTestFunc(purpose="测试 fs 库卡方计算及计算时间")
def test_fs_chi2():
    tasks = []
    for feature_name in feature_name_list:
        tasks.append(
            filter_methods.task_chi2.remote(ray_data[label_name],
                                            ray_data[feature_name].values.astype(dtype=np.float32),
                                            feature_name))
    chi2_v = [ret[1] if ret[1] is not None else -1 for ret in ray.get(tasks)]
    print("fs 卡方值：{}".format(chi2_v))
    fs_feature_rank[SupportMethods.CHI2.name] = np.argsort(np.array(chi2_v))[::-1]
    fs_feature_value[SupportMethods.CHI2.name] = np.array(chi2_v)


@WrapTestFunc(purpose="测试 sk 库互信息计算及计算时间")
def test_sk_mutual_info():
    mi = mutual_info_classif(data.drop(columns=label_name), data[label_name], random_state=42)
    print("sk 互信息值：\n{}".format(mi))
    sk_feature_rank[SupportMethods.MUTUAL_INFO.name] = np.argsort(mi)[::-1]
    sk_feature_value[SupportMethods.MUTUAL_INFO.name] = mi


@WrapTestFunc(purpose="测试 fs 库互信息计算及计算时间")
def test_fs_mutual_info():
    tasks = []
    for feature_name in feature_name_list:
        tasks.append(
            filter_methods.task_mutual_info.remote(ray_data[label_name],
                                                   ray_data[feature_name].values,
                                                   feature_name, y_discrete=True, random_state=42))
    mi_list = [ret[1] for ret in ray.get(tasks)]
    print("fs 互信息值：\n{}".format(mi_list))
    fs_feature_rank[SupportMethods.MUTUAL_INFO.name] = np.argsort(np.array(mi_list))[::-1]
    fs_feature_value[SupportMethods.MUTUAL_INFO.name] = np.array(mi_list)


@WrapTestFunc(purpose="测试 lightgbm 特征重要性及计算时间")
def test_lgb():
    lgb = LGBMClassifier(objective="binary", boosting_type='gbdt', num_leaves=32, max_depth=5,
                         min_split_gain=0, min_child_weight=1e-3, min_child_samples=10, subsample=1.0,
                         subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, silent=True, learning_rate=0.1,
                         reg_lambda=0.0, random_state=None, importance_type="split")
    lgb.fit(data.drop(columns=label_name), data[label_name], eval_set=(data.drop(columns=label_name), data[label_name]),
            eval_metric="binary_error", early_stopping_rounds=None,
            init_score=None, verbose=True)
    print("light_gbm 特征重要性：iteration=None \n{}".format(lgb.feature_importances_))
    print("light_gbm 特征排序（按重要性从大到小）：\n{}".format(np.argsort(lgb.feature_importances_)[::-1]))
    sk_feature_rank[SupportMethods.FEATURE_IMPORTANCE.name] = np.argsort(lgb.feature_importances_)[::-1]
    sk_feature_value[SupportMethods.FEATURE_IMPORTANCE.name] = lgb.feature_importances_
    # shap_values = shap.TreeExplainer(lgb.booster_).shap_values(data.drop(columns=label_name), data[label_name])
    # print("shape of shap value:{}".format(shap_values[0].shape))
    # shap.summary_plot(shap_values, data.drop(columns=label_name), feature_name_list, title="light_gbm")
    global lgb_model
    lgb_model = lgb


@WrapTestFunc(purpose="测试 ray_lightgbm 特征重要性及计算时间")
def test_lgb_ray():
    ray_d_matrix = RayDMatrix(ray_data, label_name)
    m, importance, lgb = ray.get(lgb_ray.remote(train_data=ray_d_matrix, num_actors=1))
    print("ray_on_light_gbm 特征重要性：\n{}".format(importance))
    print("ray_on_light_gbm 特征排序（按重要性从大到小）：\n{}".format(np.argsort(importance)[::-1]))
    print("对应特征名：{}".format(np.array(feature_name_list)[np.argsort(importance)[::-1]]))
    fs_feature_rank[SupportMethods.FEATURE_IMPORTANCE.name] = np.argsort(importance)[::-1]
    fs_feature_value[SupportMethods.FEATURE_IMPORTANCE.name] = importance
    # shap_values = shap.TreeExplainer(lgb.booster_).shap_values(data.drop(columns=label_name), data[label_name])
    # print("shap_values: {}".format(shap_values))
    # shap.summary_plot(shap_values, data.drop(columns=label_name), feature_name_list, title="light_gbm_on_ray")
    global lgb_on_ray_model
    lgb_on_ray_model = lgb


@WrapTestFunc(purpose="测试 sk stepwise及计算时间")
def test_sk_rfe():
    rfe = RFE(estimator=lgb_model, n_features_to_select=5, step=2, verbose=1)
    rfe.fit(data.drop(columns=label_name), data[label_name])
    print("sk 特征ranking: {}".format(rfe.ranking_))
    sk_feature_rank[SupportMethods.STEP_WISE.name] = np.argsort(rfe.ranking_)
    sk_feature_value[SupportMethods.STEP_WISE.name] = rfe.ranking_


@WrapTestFunc(purpose="测试 fs stepwise及计算时间")
def test_fs_rfe():
    method, feature_rank = ray.get(
        recursive_feature_elimination.remote(ray_data[label_name], ray_data.drop(columns=label_name),
                                             lgb_on_ray_model, n_features_to_select=5,
                                             step=2, verbose=1, num_actors=1,
                                             importance_getter="auto", cpus_per_actor=2)
    )
    print("fs 特征ranking: {}".format(feature_rank))
    fs_feature_rank[SupportMethods.STEP_WISE.name] = np.argsort(np.array(feature_rank))
    fs_feature_value[SupportMethods.STEP_WISE.name] = np.array(feature_rank)


@WrapTestFunc(purpose="测试 sk Permutation importance 及计算时间")
def test_sk_permutation_imp():
    result = permutation_importance(
        lgb_model, data.drop(columns=label_name), data[label_name], n_repeats=2, random_state=42
    )
    print("sk permutation importance:\n{}".format(result.importances_mean))
    pi_sorted_idx = result.importances_mean.argsort()
    sk_feature_rank[SupportMethods.PERMUTATION_IMPORTANCE.name] = pi_sorted_idx[::-1]
    sk_feature_value[SupportMethods.PERMUTATION_IMPORTANCE.name] = result.importances_mean

    # print("After permutation top 50 important features: {}".format(pi_sorted_idx[-50:][::-1]))
    print("sk_feature_rank:{}".format(sk_feature_rank))
    # print("fs_feature_rank:{}".format(fs_feature_rank))


@WrapTestFunc(purpose="测试 fs Permutation importance 及计算时间")
def test_fs_permutation_imp():
    """
    lightgbm on ray 在每次预测时会开启 N 个actor, 而
    Returns:

    """
    fs_result = fs_pi(
        lgb_on_ray_model.booster_, ray_data.drop(columns=label_name), ray_data[label_name], n_repeats=2,
        random_state=42, n_jobs=3
    )
    print("fs permutation importance:\n{}".format(fs_result.importances_mean))
    pi_sorted_idx = fs_result.importances_mean.argsort()
    fs_feature_rank[SupportMethods.PERMUTATION_IMPORTANCE.name] = pi_sorted_idx[::-1]
    fs_feature_value[SupportMethods.PERMUTATION_IMPORTANCE.name] = fs_result.importances_mean
    print("fs_feature_rank:{}".format(fs_feature_rank))


@WrapTestFunc(purpose="对比各项计算结果是否相同或相似", timeit=False)
def test_result():
    """
    测试通过标准：
    除空值率外，
    Returns:

    """
    for v in SupportMethods.__members__:
        if v == "NULL_FILTER":
            continue
        try:
            if sk_feature_rank[v].tolist() == fs_feature_rank[v].tolist():
                print("方法：{} 计算的特征选择结果完全相同".format(v))
            elif sum(sk_feature_rank[v] == fs_feature_rank[v]) / len(feature_name_list) >= 0.8:
                print("方法：{} 计算的特征选择结果 80% 相同".format(v))
                print("sk 库特征在方法：{} 中的排序结果：{}\n"
                      "fs 库在特征方法：{} 中的排序结果：{}".format(v, sk_feature_rank[v], v, fs_feature_rank[v]))
            else:
                # 如果排序结果不同，检查值相差是否在0.1内，如果是则认为相同
                print("方法：{} 计算的特征选择结果不同".format(v))
                print("\t\tsk 库特征在方法：{} 中的排序结果：{}\n"
                      "\t\tfs 库在特征方法：{} 中的排序结果：{}".format(v, sk_feature_rank[v], v, fs_feature_rank[v]))
        except KeyError as e:
            print("方法：{} 未计算成功！".format(v))
        print("-" * 20)

"""
@WrapTestFunc(purpose="测试fs 库进行特征选择后的效果，分布选择40%，60%，80%")
def test_fs_selected_effect():
    rate = [0.4, 0.6, 0.8]
    # 1. 根据排序选择前 百分之N 的特征
    # 2. 训练简单的模型，与完整特征效果对比
    pass"""


if __name__ == '__main__':
    # test_null_rate()
    # pytest.main(['-s', 'test_compare_fs_sk.py'])
    # test_toad_iv()
    # test_sk_variance()
    # test_fs_variance()
    # test_sk_chi2()
    # test_fs_chi2()
    # test_scipy_pearson()
    # test_fs_pearson()
    # test_fs_iv()
    # test_result()
    # test_scorecard_iv()
    # test_toad_iv()
    test_lgb()
    test_lgb_ray()
    test_sk_permutation_imp()
    test_fs_permutation_imp()
    # test_sk_mutual_info()
    # test_fs_mutual_info()
    # test_sk_rfe()
    # test_fs_rfe()
    print("sk 库特征排序：{}".format(sk_feature_rank))
    print("fs 库特征排序：{}".format(fs_feature_rank))
    ray.shutdown()
# [489 382  21  64 184 126 151 172   0   0 172  91 226   0 197 115 128 104 378]
