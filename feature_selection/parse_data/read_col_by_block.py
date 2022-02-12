import ray
from pyarrow import csv
from typing import List, Any, Dict, Union, Optional, Tuple, Callable, \
    TypeVar, TYPE_CHECKING
from ray.data import Dataset
from ray.data.impl.arrow_block import ArrowRow
from tqdm import tqdm
import time
import numpy as np
import pyarrow
import sys

from parse_data.read_api import FSRayDataSets


def get_features_data(datasets: Dataset[ArrowRow], *, features: Union[str, List[str]] = None,
                      zero_copy_only=False, remote=True, cal_null_rate=False):
    start = time.time()
    if remote:
        all_data_of_feature: List[List, List] = ray.get(
            [aggregate_col_parallel.remote(block, schemas=features,
                                           zero_copy_only=zero_copy_only,
                                           cal_null_rate=cal_null_rate) for block in
             datasets.get_blocks()])
        print("read feature data duration with parallel:{}".format(time.time() - start))
    else:
        all_data_of_feature = aggregate_col_blocking(datasets, schemas=features,
                                                     zero_copy_only=zero_copy_only, cal_null_rate=cal_null_rate)
        print("read feature data duration blocking:{}".format(time.time() - start))
    start = time.time()
    # all_data_of_feature_label = np.concatenate(all_data_of_feature, axis=1)
    all_data_of_feature_label = np.concatenate([all_data_of_feature[i][0] for i in range(len(all_data_of_feature))],
                                               axis=1)
    print("concatenate cost:{}".format(time.time() - start))

    null_data_count = np.sum([all_data_of_feature[i][1] for i in range(len(all_data_of_feature))], axis=0)
    print("空值率：{}".format(null_data_count))
    print("all_data_of_feature_label-size:{} M, shape:{}".format(sys.getsizeof(all_data_of_feature_label) / 1000 / 1024,
                                                                 all_data_of_feature_label.shape))
    return all_data_of_feature_label


@ray.remote
def aggregate_col_parallel(block: pyarrow.lib.Table, *,
                           schemas: Union[str, List[str]], zero_copy_only=False, cal_null_rate=False):
    # print("block type:{}".format(type(block)))
    block_data = block
    rows = block_data.num_rows
    if schemas is None:
        schemas = block_data.schema.names[:10]
    if schemas == "all":
        schemas = block_data.schema.names
    feature_data = []
    feature_null_data_count = []
    for feature_name in tqdm(schemas):
        feature_column_data: pyarrow.lib.ChunkedArray = block_data[feature_name]
        feature_col_flatten_data = feature_column_data.combine_chunks()
        feature_col_np_data: np.ndarray = feature_col_flatten_data.to_numpy(zero_copy_only=zero_copy_only)

        if cal_null_rate:
            if isinstance(feature_col_flatten_data, pyarrow.StringArray):
                feature_null_count = rows - len(feature_col_np_data)
                for i in range(len(feature_col_np_data)):
                    try:
                        if np.isnan(float(feature_col_np_data[i])):
                            feature_null_count += 1
                    except ValueError as e:
                        # 非 数值型数据
                        # print("非数值型数据：{}".format(feature_col_np_data[i]))
                        feature_col_np_data[i] = np.nan
                        feature_null_count += 1
            else:
                feature_null_count = feature_col_flatten_data.null_count

            print("列:{} 空值个数：{}， 数据类型：{}"
                  .format(feature_name, feature_null_count, type(feature_col_flatten_data)))
            feature_null_data_count.append(feature_null_count)
        feature_data.append(feature_col_np_data)
        del feature_col_np_data
    return feature_data, feature_null_data_count


def aggregate_col_blocking(datasets: Dataset[ArrowRow], *, schemas: Union[str, List[str]] = None,
                           zero_copy_only=False, cal_null_rate=False):
    all_block_data = []
    if schemas is None:
        schemas = datasets.schema().names[:10]
    if schemas == "all":
        schemas = datasets.schema().names
    for block in datasets.get_blocks():
        block_feature_data = []
        block_null_count = []
        print("开始对block:{}进行读取：".format(block))
        block_data: pyarrow.lib.Table = ray.get(block)
        rows = block_data.num_rows
        for feature_name in tqdm(schemas):
            feature_column_data: pyarrow.lib.ChunkedArray = block_data[feature_name]
            feature_col_flatten_data = feature_column_data.combine_chunks()
            feature_col_np_data: np.ndarray = feature_col_flatten_data.to_numpy(zero_copy_only=zero_copy_only)

            if cal_null_rate:
                if isinstance(feature_col_flatten_data, pyarrow.StringArray):
                    feature_null_count = rows - len(feature_col_np_data)
                    for i in range(len(feature_col_np_data)):
                        try:
                            if np.isnan(float(feature_col_np_data[i])):
                                feature_null_count += 1
                        except ValueError as e:
                            # 非 数值型数据
                            feature_col_np_data[i] = np.nan
                            feature_null_count += 1
                else:
                    feature_null_count = feature_col_flatten_data.null_count
                print("列:{} 空值个数：{}， 数据类型：{}"
                      .format(feature_name, feature_null_count, type(feature_col_flatten_data)))
                block_null_count.append(feature_null_count)

            block_feature_data.append(feature_col_np_data)
        print("完成block:{}数据读取".format(block))
        all_block_data.append([block_feature_data, block_null_count])
        del block_data
        del block_feature_data
        del block_null_count
        # print(all_data_of_feature[-1][-1].shape)
    return all_block_data

@ray.remote
def read_block(block, feature_name, zero_copy_only):
    print("read block:{}".format(block))
    feature_column_data: pyarrow.lib.ChunkedArray = block[feature_name]
    feature_col_flatten_data = feature_column_data.combine_chunks()
    feature_col_np_data: np.ndarray = feature_col_flatten_data.to_numpy(zero_copy_only=zero_copy_only)
    return feature_col_np_data

@ray.remote
def get_feature_data_remote(datasets, schemas, zero_copy_only=True):
    all_block_data = []
    for block in datasets.get_blocks():
        block_feature_data = []
        block_null_count = []
        print("开始对block:{}进行读取：".format(block))
        for feature_name in tqdm(schemas):
            """feature_column_data: pyarrow.lib.ChunkedArray = block[feature_name]
            feature_col_flatten_data = feature_column_data.combine_chunks()
            feature_col_np_data: np.ndarray = feature_col_flatten_data.to_numpy(zero_copy_only=zero_copy_only)"""
            feature_col_np_data = ray.get(read_block.remote(block, feature_name, zero_copy_only))
            block_feature_data.append(feature_col_np_data)
        print("完成block:{}数据读取".format(block))
        all_block_data.append(block_feature_data)
        del block_feature_data
        del block_null_count
        # print(all_data_of_feature[-1][-1].shape)
    return all_block_data


class GetBlockColDataRemote:
    def __init__(self, block):
        self.block = block

    def read_block_data(self, schemas, zero_copy_only):
        print("type block:{}".format(type(self.block)))
        block_feature_data = []
        for feature_name in tqdm(schemas):
            feature_col_np_data: np.ndarray = self.block[feature_name].combine_chunks().to_numpy(zero_copy_only=zero_copy_only)
            block_feature_data.append(feature_col_np_data)
        return block_feature_data


class GetFeatureDataRemote:
    def __init__(self, ds, schemas, zero_copy_only=True):
        self.ds = ds
        self.schemas = schemas
        self.zero_copy_only = zero_copy_only
        self.all_block_data = []

    def read_each_block(self):
        block_reader = ray.remote(GetBlockColDataRemote)
        for block in self.ds._blocks:
            print("开始对block:{}进行读取：".format(block))
            block_reader_actor = block_reader.remote(block)
            read_block_worker = block_reader_actor.read_block_data.remote(self.schemas, self.zero_copy_only)
            feature_col_np_data = ray.get(read_block_worker)
            ray.kill(block_reader_actor)

            print("type feature_col_np_data: {}".format(type(feature_col_np_data)))
            print("完成block:{}数据读取".format(block))
            self.all_block_data.append(feature_col_np_data)
            del feature_col_np_data
            print("当前all_block_data 长度：{}，大小：{}".format(len(self.all_block_data), sys.getsizeof(self.all_block_data)))
        return self.all_block_data


if __name__ == '__main__':
    my_ds = FSRayDataSets("label")
    # dataset = ds.read_data(["D:\\data\\ray_lr\\mnist_dataset_csv\\mnist_train_norm.csv"] * 2) dataset =
    # ds.read_data("D:\\data\\ray_lr\\mnist_dataset_csv\\mnist_train_norm.csv") read_option = csv.ReadOptions(
    # column_names=["SeriousDlqin2yrs","RevolvingUtilizationOfUnsecuredLines","age",
    # "NumberOfTime30-59DaysPastDueNotWorse","DebtRatio","MonthlyIncome","NumberOfOpenCreditLinesAndLoans",
    # "NumberOfTimes90DaysLate","NumberRealEstateLoansOrLines","NumberOfTime60-89DaysPastDueNotWorse",
    # "NumberOfDependents"]) read_options = {"read_options": read_option} 指定读取CSV文件哪些列
    convert_option = csv.ConvertOptions(include_columns=["SeriousDlqin2yrs","RevolvingUtilizationOfUnsecuredLines","age","NumberOfTime30-59DaysPastDueNotWorse","DebtRatio","MonthlyIncome","NumberOfOpenCreditLinesAndLoans","NumberOfTimes90DaysLate","NumberRealEstateLoansOrLines","NumberOfTime60-89DaysPastDueNotWorse","NumberOfDependents"])
    read_options = {"convert_options": convert_option}
    dataset = my_ds.read_data("D:\\data\\kaggle\\giveMeSomeCredit\\cs-training.csv", **read_options)
    # dataset = ds.read_data("../data/test_nan_data.csv")
    print(dataset.num_blocks())
    schemas = dataset.schema().names
    # get_features_data(dataset, zero_copy_only=True, remote=False)
    # get_features_data(dataset, features="all", zero_copy_only=False, cal_null_rate=True, remote=False)
    # col_data = ray.get(get_feature_data_remote.remote(dataset, dataset.schema().names, zero_copy_only=False))
    feature_data_remote = ray.remote(GetFeatureDataRemote)
    feature_data_actor = feature_data_remote.remote(dataset, schemas, zero_copy_only=False)
    all_block_feature_data = ray.get(feature_data_actor.read_each_block.remote())
    print("-" * 20)
    # get_features_data(dataset, zero_copy_only=False, cal_null_rate=True, remote=False)
    # print(get_features_data(dataset, features=["feature4"], zero_copy_only=True, remote=False))
    # print(dataset)
    # dataset = ds.read_data("D:\\data\\ray_lr\\mnist_dataset_csv\\mnist_test_x.npy")
    # print(dataset)
    ds = dataset.take(10)

    for d in ds:
        print(d)
