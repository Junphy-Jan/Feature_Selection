"""过滤式特征选择"""
# Authors: 21080443
from typing import List, Union
import numpy as np
import ray
from auto_feature_select.embedded_methods import EmbeddedActor
from auto_feature_select.filter_methods import FilterActor
from auto_feature_select.wrapper_methods import WrapperActor
from fs_util.func import SupportMethods
import time
from parse_data.read_api import FSRayDataSets

__FILTER_METHODS__ = [SupportMethods.VARIANCE.name, SupportMethods.PEARSON_CORRELATION.name, SupportMethods.CHI2.name,
                      SupportMethods.IV.name, SupportMethods.NULL_FILTER.name, SupportMethods.MUTUAL_INFO.name]


@ray.remote(num_cpus=1)
class FeatureSelection:
    def __init__(self, data: Union[str, List[str]], label_col_name: str, objective: str = "binary", *,
                 select_n: Union[float, int] = 0.5,
                 nan_replace: Union[str, float] = "avg", null_rate_limit=0.3, include_col: List[str] = None,
                 methods: List[str] = None, num_actors: int = 2, cpu_per_actor: int = 2, n_repeats: int = 2,
                 step: int = 1, partition: float = 0.2, n_jobs=2):
        """
        特征选择调度函数

        Args:
            data: 数据
            label_col_name: 标签列名
            objective: 分类任务或回归任务，对["STEP_WISE", "FEATURE_IMPORTANCE", "PERMUTATION_IMPORTANCE"] 三种方法有效，
                       可设置的值："binary", "multiclass", "regression"，分别对应 二分类，多分类，回归任务
            select_n: 选择N个标签，如果为浮点数，则选择对应的百分比个，通过int()取整；如果为整数，则取N个
            nan_replace: 空值替换方式，如果为数字，则替换为该数字，如果为字符串，可选择："avg": 列数据平均值填充
            null_rate_limit: 特征数据最低空值率，超过此比例，将在此后的计算中排除此特征
            include_col: 最终选择的特征必须包含的特征名，计算时
            methods: 方法名列表，支持的方法名：["VARIANCE", "PEARSON_CORRELATION", "CHI2", "IV", "NULL_FILTER",
                     "MUTUAL_INFO", "STEP_WISE", "FEATURE_IMPORTANCE", "PERMUTATION_IMPORTANCE"],
                     默认为None，会选择 [VARIANCE, PEARSON_CORRELATION, CHI2, IV, NULL_FILTER, MUTUAL_INFO] 6种过滤式方法。
                     STEP_WISE, FEATURE_IMPORTANCE, PERMUTATION_IMPORTANCE 方法耗时相对较长，且需要训练模型。
                     选择 'STEP_WISE' 或 'PERMUTATION_IMPORTANCE' 都会以树模型为基础，
                     即选择此两种方法都会有 'FEATURE_IMPORTANCE'。
                     若选择所有方法，可填 ["all"]，任意选择一种过滤式方法均会通过 空值过滤器，即"NULL_FILTER"
            num_actors: 对["STEP_WISE", "FEATURE_IMPORTANCE", "PERMUTATION_IMPORTANCE"] 三种方法有效，用于指定训练树模型
                        的进程数目，最小值为2
            cpu_per_actor: 对["STEP_WISE", "FEATURE_IMPORTANCE", "PERMUTATION_IMPORTANCE"] 三种方法有效，用于指定训练树模型
                           的CPU数目，最小值为2
            n_repeats: 对 PERMUTATION_IMPORTANCE 有效，对某列特征 置换 N 次计算平均分
            step: 对 STEP_WISE 有效，每一步减少 N 个特征
            partition: 对 PERMUTATION_IMPORTANCE 有效，使用全部数据的比例，默认为0.2
            n_jobs: 置换重要性中的并行数，如果数据量较大，可调低此参数
        """
        # ray.init()
        self.fs_ds = FSRayDataSets(label_col_name=label_col_name, data_source=data)
        self.schemas = self.fs_ds.schemas
        # 转换为方法列表
        if methods is None:
            self.methods = __FILTER_METHODS__
        elif methods == ["all"]:
            self.methods = [v.name for v in SupportMethods]
        elif not isinstance(methods, List):
            raise ValueError("需要正确指定 'methods' 参数！当前值：{}".format(methods))
        # 如果包含 stepwise 或 包含 permutation_importance 不包含 feature_importance，需要先训练一个基线模型
        elif (methods.__contains__(SupportMethods.STEP_WISE.name) or
              methods.__contains__(SupportMethods.PERMUTATION_IMPORTANCE.name)) \
                and not methods.__contains__(SupportMethods.FEATURE_IMPORTANCE.name):
            print("将添加：{} 方法".format(SupportMethods.FEATURE_IMPORTANCE.name))
            methods.append(SupportMethods.FEATURE_IMPORTANCE.name)
            self.methods = methods
        else:
            self.methods = methods

        if isinstance(data, str):
            data = [data]
        if num_actors > len(data) and self.methods.__contains__(SupportMethods.FEATURE_IMPORTANCE.name):
            print("训练actor数需小于等于文件数，actor 将被设置为：{}".format(len(data)))
            num_actors = len(data)
        self.select_n = select_n
        # 分类或回归
        self.objective = objective
        self.filter_actor = None
        self.wrapper_actor = None
        self.embedded_actor = None
        # 转换为选择个数
        if isinstance(self.select_n, float):
            self.select_n = int(select_n * len(self.schemas))
        if self.select_n <= 0:
            print("选择的比例：{} 太小，无法选择一个特征！".format(select_n))
            return
        if self.select_n >= len(self.schemas):
            print("要选择的特征数目：{} >= 所有特征之和，无需选择！".format(select_n))
            return

        available_resources = ray.available_resources()
        print("可用资源情况：{}".format(available_resources))
        self.cpu_num = int(available_resources["CPU"])
        self.nan_replace = nan_replace
        self.null_rate_limit = null_rate_limit
        self.include_col = include_col
        self.num_actors = num_actors
        self.cpus_per_actor = cpu_per_actor
        self.n_repeats = n_repeats
        self.step = step
        self.partition = partition

        self.label_data = None
        self.feature_data = {}

        # 训练好的树模型
        self.estimator = None
        # 特征排序
        self.feature_rank = []
        # 置换重要性中的并行数，如果数据量较大，可调低此参数
        self.n_jobs = n_jobs

    def fetch_data(self, feature_name: str):
        if feature_name == self.fs_ds.label_col_name:
            self.label_data = self.fs_ds.get_column_by_name_from_source(feature_name, transformer="modin")
        else:
            self.feature_data[feature_name] = self.fs_ds.get_column_by_name_from_source(feature_name,
                                                                                        transformer="modin")
            print("从dataset 获取了特征：{}的数据：{}".format(feature_name, self.feature_data[feature_name].shape))

    def handle_filter(self, feature_name, x_discrete="auto", y_discrete=True):
        # 过滤式
        if self.filter_actor is not None and self.feature_data[feature_name] is not None:
            filter_future = self.filter_actor.work.remote(label_data=self.label_data,
                                                          feature_data=self.feature_data[feature_name],
                                                          feature_name=feature_name, nan_replace=self.nan_replace,
                                                          null_rate_limit=self.null_rate_limit,
                                                          n_neighbors=3, x_discrete=x_discrete, y_discrete=y_discrete)

            # ray.get(filter_future)
            # print("handle filter 处理特征：{} 完成！".format(feature_name))
            # self.feature_data[feature_name] = None
            return filter_future
        else:
            print("特征数据：{} 为空".format(feature_name))

    def handle_wrapper(self, estimator):
        # 包裹式
        if self.wrapper_actor is not None:
            return ray.get(
                self.wrapper_actor.work.remote(self.fs_ds.read_data(transformer="modin", keep_ref=True), estimator,
                                               self.fs_ds.label_col_name,
                                               schemas=self.schemas, step=self.step, verbose=1,
                                               importance_getter="auto", num_actors=self.num_actors,
                                               cpus_per_actor=self.cpus_per_actor))
        else:
            print("请先指定包裹式actor！")

    def handle_embedded(self, train_set, label_col_name, objective, schemas, num_actors, cpus_per_actor,
                        n_jobs, n_repeats):
        # 嵌入式
        if self.embedded_actor is not None:
            embedded_future = self.embedded_actor.work.remote(train_set, objective=objective,
                                                              schemas=schemas, label_col_name=label_col_name,
                                                              num_actors=num_actors, cpus_per_actor=cpus_per_actor,
                                                              n_jobs=n_jobs, n_repeats=n_repeats,
                                                              partition=self.partition)
            return ray.get(embedded_future)
        else:
            print("未设置对应的 embedded_actor！")

    def select(self):
        # 过滤式
        if len(set(self.methods) & set(__FILTER_METHODS__)) >= 1:
            self.filter_actor = FilterActor.remote("filter_actor", self.methods)
            print("-" * 20 + "开始过滤式任务" + "-" * 20)
            f_start = time.time()
            # 获取标签数据
            self.fetch_data(self.fs_ds.label_col_name)
            features = self.schemas
            # 获取特征数据
            self.fetch_data(features[0])
            for i in range(0, len(features)):
                filter_handle_remote = self.handle_filter(features[i])
                if i != len(features) - 1:
                    self.fetch_data(features[i + 1])
                ray.get(filter_handle_remote)
                print("handle filter 处理特征：{} 完成！".format(features[i]))

            self.feature_rank.append(ray.get(self.filter_actor.get_attr.remote()))
            print("过滤式方法对特征的处理结果：\n{}".format(self.feature_rank))
            ray.kill(self.filter_actor)
            del self.filter_actor
            del self.feature_data
            print("-" * 20 + "结束过滤式任务! 用时：{}".format(time.time() - f_start) + "-" * 20)

        # 过滤式
        if self.methods.__contains__(SupportMethods.FEATURE_IMPORTANCE.name):
            self.embedded_actor = EmbeddedActor.remote("embedded_actor", self.methods)
            e_start = time.time()
            print("-" * 20 + "开始嵌入式任务" + "-" * 20)
            self.handle_embedded(self.fs_ds.read_data(transformer="modin", keep_ref=True), self.fs_ds.label_col_name,
                                 objective=self.objective, schemas=self.schemas, num_actors=self.num_actors,
                                 cpus_per_actor=self.cpus_per_actor, n_jobs=self.n_jobs, n_repeats=self.n_repeats)
            # 此处get 出来时为了 kill 掉 embedded actor，模型文件不大
            self.estimator = ray.get(self.embedded_actor.get_estimator.remote())
            self.feature_rank.append(ray.get(self.embedded_actor.get_attr.remote()))
            ray.kill(self.embedded_actor)
            print("-" * 20 + "结束嵌入式任务! 用时：{}".format(time.time() - e_start) + "-" * 20)

        # 包裹式
        if self.methods.__contains__(SupportMethods.STEP_WISE.name):
            if self.estimator is None:
                raise ValueError("请先待树模型训练完成！")
            w_start = time.time()
            print("-" * 20 + "开始包裹式任务" + "-" * 20)
            self.estimator = ray.put(self.estimator)
            self.wrapper_actor = WrapperActor.remote("wrapper", select_best_n=self.select_n)
            self.handle_wrapper(self.estimator)
            self.feature_rank.append(ray.get(self.wrapper_actor.get_attr.remote()))
            ray.kill(self.wrapper_actor)
            print("-" * 20 + "结束包裹式任务! 用时：{}".format(time.time() - w_start) + "-" * 20)

        del self.fs_ds

    def get_feature_rank(self):
        return self.feature_rank

    def getattr(self, attr):
        return self.__getattribute__(attr)


def parse_feature_rank(feature_rank, schemas):
    """

    Args:
        feature_rank: 从各种方法获得的每个特征对应每个方法的数据 e.g. {"feature0":{"NULL_FILTER":0.0, "IV":1.1}, "feature1":{...}}
        schemas: 特征列表

    Returns: method_rank, method_value

    """
    # 每个方法对应每个特征的分数
    method_value = {}
    method_rank = {}
    for d in feature_rank:
        for schema in schemas:
            feature_method_value = d[schema]
            for method in feature_method_value:
                if not method_value.__contains__(method):
                    method_value[method] = [feature_method_value[method]]
                else:
                    method_value[method].append(feature_method_value[method])

    for method in method_value:
        # 按从小到大顺序排
        if method == SupportMethods.NULL_FILTER.name or method == SupportMethods.STEP_WISE.name:
            rank_list = np.array(method_value[method]).argsort().tolist()
        else:
            rank_list = np.array(method_value[method]).argsort()[::-1].tolist()
        # 解决相同值其排序值不同
        method_rank[method] = rank_list
    return method_rank, method_value


def select_best_n(best_n, method_rank, method_value):
    """

    Args:
        best_n: 要选择的个数
        method_rank: 每个方法对应的特征排序，e.g. {'CHI2':{0,2,4,3,1}, 'VARIANCE':{1,0,2,3,4}}
        method_value: 每个方法对应的特征得分. e.g. {'CHI2':{102.2, 9.31, 88, 63.78, 70.9},
                                                 'VARIANCE':{17.57, 166182, 17.38, 17.26, 1.22}}

    Returns: 每种方法选出的前N个特征序号。如得分相同，则某种方法实际选出的个数可能大于N

    """
    selected_schema_pos = []
    for v in method_rank.keys():
        method = v
        rank = method_rank[method]
        # 当前方法前N个特征 index
        selected_pos = rank[:best_n]
        i = best_n
        while i < len(rank) - 1:
            last_rank_pos = rank[i - 1]
            last_rank_pos_plus1 = rank[i]
            if method_value[method][last_rank_pos_plus1] == method_value[method][last_rank_pos]:
                selected_pos.append(rank[i])
                i += 1
            else:
                break
        selected_schema_pos.append(selected_pos)
    return selected_schema_pos


def select_feature(data: Union[str, List[str]], label_col_name: str, *, objective: str = "binary",
                   cluster_address: str = None, select_n: Union[float, int] = 0.5, cpu_per_actor: int = 2,
                   nan_replace: Union[str, float] = "avg", null_rate_limit=0.3, num_actors: int = 2,
                   include_col: List[str] = None, methods: List[str] = None, n_jobs: int = 2, n_repeats: int = 2,
                   step: int = 1, partition: float = 0.2):
    """
    特征选择

    Args:
        data: 需要进行特征选择的数据，数据路径，如：["../train1.csv", "../train2.csv"]
        label_col_name: 标签列名
        objective: 分类任务或回归任务，对["STEP_WISE", "FEATURE_IMPORTANCE", "PERMUTATION_IMPORTANCE"] 三种方法有效，
                   可设置的值："binary", "multiclass", "regression"，分别对应 二分类，多分类，回归任务
        cluster_address: 集群地址，默认为None
        select_n: 选择的特征个数，如果是整型，则选择 N 个，否则选择前 N 比例
        nan_replace: 空值替换值，"avg" 或 固定的值，如果是 "avg" 则会选择该列的平均值
        null_rate_limit: 限制空值比例，如果空值率大于此则不会在过滤类方法种计算此列数据
        include_col: 必选的列名
        methods: 方法名列表，支持的方法名：["VARIANCE", "PEARSON_CORRELATION", "CHI2", "IV", "NULL_FILTER",
                 "MUTUAL_INFO", "STEP_WISE", "FEATURE_IMPORTANCE", "PERMUTATION_IMPORTANCE"],
                 默认为None，会选择 [VARIANCE, PEARSON_CORRELATION, CHI2, IV, NULL_FILTER, MUTUAL_INFO] 6种过滤式方法。
                 STEP_WISE, FEATURE_IMPORTANCE, PERMUTATION_IMPORTANCE 方法耗时相对较长，且需要训练模型。
                 选择 'STEP_WISE' 或 'PERMUTATION_IMPORTANCE' 都会以树模型为基础，
                 即选择此两种方法都会有 'FEATURE_IMPORTANCE'。
                 若选择所有方法，可填 ["all"]，任意选择一种过滤式方法均会通过 空值过滤器，即"NULL_FILTER"
        num_actors: 对["STEP_WISE", "FEATURE_IMPORTANCE", "PERMUTATION_IMPORTANCE"] 三种方法有效，用于指定训练树模型
                        的进程数目，最小值为2
        cpu_per_actor: 对["STEP_WISE", "FEATURE_IMPORTANCE", "PERMUTATION_IMPORTANCE"] 三种方法有效，用于指定训练树模型
                       的CPU数目，最小值为2
        n_repeats: 对 PERMUTATION_IMPORTANCE 有效，对某列特征 置换 N 次计算平均分
        step: 对 STEP_WISE 有效，每一步减少 N 个特征
        partition: 对 PERMUTATION_IMPORTANCE 有效，使用全部数据的比例，默认为0.2
        n_jobs: 置换重要性中的并行数，如果数据量较大，可调低此参数

    Returns:返回选择的特征以及特征得分

    """
    ray.shutdown()
    if cluster_address is None:
        ray.init()
    else:
        ray.init("ray://" + cluster_address)

    fs = FeatureSelection.remote(data, label_col_name, objective=objective, select_n=select_n, nan_replace=nan_replace,
                                 null_rate_limit=null_rate_limit, include_col=include_col, methods=methods,
                                 num_actors=num_actors, cpu_per_actor=cpu_per_actor, n_repeats=n_repeats, step=step,
                                 partition=partition, n_jobs=n_jobs)
    select_future = fs.select.remote()
    ray.get(select_future)
    feature_rank = ray.get(fs.getattr.remote("feature_rank"))
    select_n = ray.get(fs.getattr.remote("select_n"))
    # 所有特征列
    schemas = ray.get(fs.getattr.remote("schemas"))
    ray.kill(fs)
    ray.shutdown()
    print("feature_rank:{}".format(feature_rank))
    method_rank, method_value = parse_feature_rank(feature_rank, schemas)
    print("method rank:{}".format(method_rank))
    print("method value:{}".format(method_value))
    selected_feature_per_method = select_best_n(select_n, method_rank=method_rank, method_value=method_value)
    print("selected_feature_per_method: {}".format(selected_feature_per_method))
    feature_score = {}
    for method_selected in selected_feature_per_method:
        for feature in method_selected:
            feature_score[feature] = feature_score.get(feature, 0) + 1
    sorted_feature_score = sorted(feature_score.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    if len(sorted_feature_score) >= select_n:
        for i in range(select_n):
            print("特征序号：{}，即特征：{}，得分：{}".format(sorted_feature_score[i][0], schemas[sorted_feature_score[i][0]],
                                                sorted_feature_score[i][1]))
        return sorted_feature_score[:select_n]
    else:
        return sorted_feature_score


if __name__ == '__main__':
    ray.shutdown()
    feature_score_ = select_feature("../data/cs-training-no-index-no-str.csv",
                                    label_col_name="SeriousDlqin2yrs", methods=["all"])
    print("feature_score{}".format(feature_score_))
