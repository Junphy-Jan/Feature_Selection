import os
from typing import List, Dict, Optional, Tuple
import numpy as np
import ray
from pyarrow.csv import ConvertOptions

__SUPPORT_FILE_FORMAT__ = {
    "csv": "csv",
    "parquet": "parquet",
    # "json": "json",
    # "npy": "npy",
}

from modin.pandas import DataFrame


class FSRayDataSets:
    def __init__(self, label_col_name):
        """
        Args:
            label_col_name: 标签名
        """
        # 标签名
        self.label_col_name = label_col_name
        self.ds = None
        # 所有特征列表
        self.schemas = None
        self.file_format = None
        self.data_source = None

    def get_data_info(self, data_source):
        self.file_format = self.__check_data_source_format(data_source)
        if isinstance(data_source, str):
            data_source = [data_source]
        self.data_source = data_source
        self.read_data(self.data_source[0])
        self.schemas = self.ds.schema().names
        self.schemas.remove(self.label_col_name)
        del self.ds
        return self.schemas

    def read_data(self, data_source, *, filesystem: Optional["pyarrow.fs.FileSystem"] = None,
                  columns: Optional[List[str]] = None, parallelism: int = 200,
                  _tensor_column_schema: Optional[Dict[str, Tuple[np.dtype, Tuple[int, ...]]]] = None,
                  transformer: str = None,
                  **read_args):
        if self.file_format is not None:
            self.file_format = self.__check_data_source_format(data_source)

        if self.file_format == __SUPPORT_FILE_FORMAT__["csv"]:
            if columns is not None:
                if read_args is None:
                    read_args = {
                        "convert_options": ConvertOptions(include_columns=columns)}
                else:
                    read_args["convert_options"] = ConvertOptions(include_columns=columns)
                ds = ray.data.read_csv(data_source, filesystem=filesystem, parallelism=parallelism, **read_args)
            else:
                ds = ray.data.read_csv(data_source, filesystem=filesystem, parallelism=parallelism, **read_args)
        elif self.file_format == __SUPPORT_FILE_FORMAT__["parquet"]:
            ds = ray.data.read_parquet(data_source, filesystem=filesystem, columns=columns, parallelism=parallelism,
                                       _tensor_column_schema=_tensor_column_schema, **read_args)
        else:
            raise NotImplementedError("数据格式：{} 暂不支持！".format(data_source))
        if self.schemas is not None:
            self.schemas = ds.schema().names
            # 移除列名中的标签列
            self.schemas.remove(self.label_col_name)
        if transformer is None:
            self.ds = ds
        elif transformer == "modin":
            self.ds = ds.to_modin()

    def __check_data_source_format(self, data_source):
        if isinstance(data_source, str):
            file_format = data_source[data_source.rindex(".") + 1:]
            try:
                return __SUPPORT_FILE_FORMAT__[file_format]
            except KeyError:
                raise ValueError("文件格式不支持，仅支持从以下文件中读取：{}".format(__SUPPORT_FILE_FORMAT__.keys()))
        if isinstance(data_source, List):
            return self.__check_data_source_format(data_source[0])
        else:
            pass

    def get_column_by_name(self, column):
        print("进程：{} 正在获取列：{} 的数据".format(os.getpid(), column))
        if self.ds is not None and isinstance(self.ds, DataFrame):
            return self.ds[column].values

    def get_column_by_name_from_data_source(self, column):
        print("进程：{} 正在获取列：{} 的数据".format(os.getpid(), column))
        if self.data_source is not None:
            print("从原始文件中读取列数据...")
            self.read_data(self.data_source, columns=[column], transformer="modin")
            return self.ds[column].values
        if self.ds is not None and isinstance(self.ds, DataFrame):
            return self.ds[column].values


if __name__ == '__main__':
    dataset = FSRayDataSets("Response")
    data_sources_ = ["../data/final_test.csv"]
    # 按列读取
    schemas_ = dataset.get_data_info(data_sources_)
    for s in schemas_:
        s_data = dataset.get_column_by_name_from_data_source(s)
        print(s.shape)

    # 全部读取
    dataset.read_data(data_sources_, transformer="modin")
    ds = dataset.ds
    print(ds.shape[0])
    print(ds.iloc[:1000, :].shape)

    col = dataset.get_column_by_name(dataset.label_col_name)
    print(len(col))

