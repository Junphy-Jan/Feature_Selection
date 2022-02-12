import os
from typing import Union, List
import logging
import ray
from modin.pandas import DataFrame
from lightgbm_ray import RayParams, train, RayLGBMClassifier, RayLGBMRegressor
# from sklearn.inspection import permutation_importance
from xgboost_ray import RayDMatrix
from auto_feature_select._base import FeatureSelectionActor
from filter_selection.inspection.fs_permutation_importance import permutation_importance
from fs_util.func import SupportMethods


# 非 sklearn 模式
def train_lightgbm(train_data, objective="binary", num_classes=2, num_actors=2):
    evals_result = {}
    metrics = []
    if objective == "regression":
        metric_error = "mse"
        metrics.append(metric_error)
    elif objective == "binary":
        metric_error = "binary_error"
        metric_log_loss = "binary_logloss"
        metrics.append(metric_error)
        metrics.append(metric_log_loss)
    elif objective == "multiclass" and num_classes > 2:
        metric_error = "multi_error"
        metric_log_loss = "multi_logloss"
        metrics.append(metric_error)
        metrics.append(metric_log_loss)
    else:
        raise ValueError("输入的objective：{}，系统支持 [regression, binary, multiclass] 其中之一".format(objective))
    params = {
        # "objective": "binary",
        "objective": objective,
        # "metric": ["binary_logloss", "binary_error"],
        "metric": metrics
    }
    if objective == "multiclass" and num_classes > 2:
        params["num_class"] = num_classes
    print("可用于训练树模型的资源：{}".format(ray.available_resources()))
    bst = train(params,
                train_data,
                evals_result=evals_result,
                valid_sets=[train_data],
                valid_names=["train"],
                ray_params=RayParams(num_actors=num_actors))
    print("训练集结果: {}".format(evals_result["train"]))
    return bst, bst.feature_importances_


@ray.remote
def lgb_ray(train_data: RayDMatrix, *, objective="binary", boosting_type='gbdt', num_leaves=32, max_depth=5,
            min_split_gain=0, min_child_weight=1e-3, min_child_samples=10, subsample=1.0,
            subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, silent=True, learning_rate=0.1,
            reg_lambda=0.0, random_state=None, importance_type="split", num_actors=2, cpus_per_actor=2):
    logging.basicConfig(level=logging.INFO)
    if objective == "binary":
        metric = "binary_error"
        lgb_classify = RayLGBMClassifier(objective=objective, boosting_type=boosting_type, num_leaves=num_leaves,
                                         max_depth=max_depth, min_split_gain=min_split_gain,
                                         min_child_weight=min_child_weight, min_child_samples=min_child_samples,
                                         subsample=subsample, subsample_freq=subsample_freq,
                                         colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, silent=silent,
                                         learning_rate=learning_rate, reg_lambda=reg_lambda,
                                         random_state=random_state, importance_type=importance_type)
    elif objective == "multi_class":
        metric = "multi_error"
        lgb_classify = RayLGBMClassifier(objective=objective)
    elif objective == "regression":
        metric = "mse"
        lgb_classify = RayLGBMRegressor(objective=objective)
    else:
        raise ValueError("objective 参数仅支持：[binary, multi_class, regression]其中之一，不支持：{}".format(objective))

    # 若数据格式是RayDMatrix，则忽略 y 参数， cpus_per_actor 最小为2
    lgb_classify.fit(X=train_data, y=None, eval_set=[(train_data, "train")], eval_metric=metric, verbose=True,
                     early_stopping_rounds=None, init_score=None,
                     ray_params=RayParams(num_actors=num_actors, cpus_per_actor=cpus_per_actor))
    # print("是否有score 方法：{}".format(getattr(lgb_classify, "score")))
    feature_importance = lgb_classify.feature_importances_
    print("light_gbm on ray 特征重要性：\n{}".format(feature_importance))
    # print("置换重要性：{}".format(task_permutation_importance(lgb_classify, valid_x, valid_y)))
    return SupportMethods.FEATURE_IMPORTANCE.name, feature_importance, lgb_classify


@ray.remote
def task_permutation_importance(estimator, permutation_x, permutation_y, *,
                                n_repeats: int = 2, random_state=None, n_jobs=None):
    print("开始计算 permutation importance")
    """
    无需训练
    """
    """
    import joblib
    from ray.util.joblib import register_ray
    register_ray()
    with joblib.parallel_backend('ray'):"""
    result = permutation_importance(
        estimator, permutation_x, permutation_y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs
    )
    # result = ray.get(result)
    print("计算 permutation importance完成")
    return SupportMethods.PERMUTATION_IMPORTANCE.name, result.importances_mean


@ray.remote
class EmbeddedActor(FeatureSelectionActor):

    def __init__(self, actor_name: str, methods_name: List[str], *, select_best_n: int = None, keep_feature: Union[str, List[str]] = None):
        super().__init__(actor_name, select_best_n, keep_feature)
        logging.basicConfig(level=logging.INFO)
        self.ret = {}
        self.estimator = None
        self.methods_name = methods_name

    def work(self, data: DataFrame, label_col_name, objective="binary", schemas: List[str] = None,
             num_actors=2, cpus_per_actor=2, n_jobs=2, n_repeats=2, partition=0.2):
        from lightgbm_ray import RayDMatrix

        print("进程：{}使用{}方法对数据类型：{}进行处理！".format(os.getpid(), self.actor_name, type(data)))
        if isinstance(data, DataFrame):
            d_matrix = RayDMatrix(data, label=label_col_name)
        else:
            raise NotImplementedError("格式：{}暂不支持".format(type(data)))
        method_fea_imp, feature_importance, estimator = ray.get(lgb_ray.remote(d_matrix, objective=objective,
                                                                               num_actors=num_actors,
                                                                               cpus_per_actor=cpus_per_actor))
        self.estimator = estimator

        avg_permutation_importance = None
        method_per_imp = SupportMethods.PERMUTATION_IMPORTANCE.name
        # 如果指定了 PERMUTATION_IMPORTANCE
        if self.methods_name.__contains__(SupportMethods.PERMUTATION_IMPORTANCE.name):
            print("开始置换重要性任务！")
            # 注意置换重要性运行时间较长，此处可选择部分数据进行重要性评估
            data_len = data.shape[0]
            partition_idx = int(data_len * partition)
            data = data.iloc[:partition_idx, :]
            print("计算特征置换重要性的数据量：{}".format(data.shape))
            method_per_imp, avg_permutation_importance = ray.get(
                task_permutation_importance.remote(estimator,
                                                   data.drop(columns=label_col_name), data[label_col_name],
                                                   n_repeats=n_repeats, n_jobs=n_jobs)
            )
            if schemas is None:
                schemas = ["feature-" + str(i) for i in range(len(feature_importance))]
            print("结束置换重要性任务！")

        if avg_permutation_importance is not None:
            for i, (fea_imp, per_imp) in enumerate(zip(feature_importance, avg_permutation_importance)):
                m_importance = {}
                m_importance[method_fea_imp] = fea_imp
                m_importance[method_per_imp] = per_imp
                self.ret[schemas[i]] = m_importance
        else:
            for i in range(len(feature_importance)):
                m_importance = {}
                m_importance[method_fea_imp] = feature_importance[i]
                self.ret[schemas[i]] = m_importance

    def get_attr(self):
        return self.ret

    def get_estimator(self):
        return self.estimator
