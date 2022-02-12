import os
from typing import Union, List

import ray
import time

from lightgbm_ray import RayParams
from modin.pandas import DataFrame

from auto_feature_select._base import FeatureSelectionActor
from filter_selection import RFE
from fs_util.func import SupportMethods


def recursive_feature_elimination(data, label_col_name, estimator, *,
                                  n_features_to_select=None, step=1, verbose=1, importance_getter="auto",
                                  num_actors=2, cpus_per_actor=2):
    # create the RFE model and select 10 attributes
    print("estimator 类型：{}".format(type(estimator)))
    rfe = RFE(estimator, n_features_to_select=n_features_to_select, step=step,
              verbose=verbose, importance_getter=importance_getter)
    ray_para = RayParams(num_actors=num_actors, cpus_per_actor=cpus_per_actor)
    rfe = rfe.fit(data, label_col_name, ray_parameter=ray_para)

    # summarize the selection of the attributes
    print("rfe 特征排序：{}".format(rfe.ranking_))
    return SupportMethods.STEP_WISE.name, rfe.ranking_
    # summarize the ranking of the attributes
    # fea_rank_ = pd.DataFrame({'cols':x.columns, 'fea_rank':rfe.ranking_})
    # fea_rank_.loc[fea_rank_.fea_rank > 0].sort_values(by=['fea_rank'], ascending = True)


@ray.remote
class WrapperActor(FeatureSelectionActor):
    def __init__(self, actor_name: str, select_best_n: int, *, keep_feature: Union[str, List[str]] = None):
        super().__init__(actor_name, select_best_n, keep_feature)
        self.ret = {}

    def work(self, data, estimator, label_col_name, *, schemas=None, step=1, verbose=1, importance_getter="auto",
             num_actors=2, cpus_per_actor=2):
        print("进程：{}使用{}方法进行递归特征消除！".format(
            os.getpid(), SupportMethods.STEP_WISE.name))
        if isinstance(data, DataFrame):
            # train_x = data.drop(columns=label_col_name)
            # print("type of train_x:{}".format(type(train_x)))
            # label = data[label_col_name]
            method, feature_rank = recursive_feature_elimination(data, label_col_name, estimator,
                                                                 n_features_to_select=self.select_best_n,
                                                                 step=step, verbose=verbose, num_actors=num_actors,
                                                                 importance_getter=importance_getter,
                                                                 cpus_per_actor=cpus_per_actor)
            for i in range(len(feature_rank)):
                self.ret[schemas[i]] = {SupportMethods.STEP_WISE.name: feature_rank[i]}
        else:
            raise NotImplementedError("格式：{}暂不支持".format(type(data)))
        # print("WrapperWorker.work: train_x.shape:{}, label.shape:{}".format(train_x.shape, label.shape))
        """
        method, feature_rank = ray.get(
            recursive_feature_elimination.remote(label, train_x, estimator, n_features_to_select=self.select_best_n,
                                                 step=step, verbose=verbose, num_actors=num_actors,
                                                 importance_getter=importance_getter, cpus_per_actor=cpus_per_actor)
        )"""


    def get_attr(self):
        return self.ret
