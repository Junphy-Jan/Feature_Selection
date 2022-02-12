import os
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import ray
from pyarrow.csv import ConvertOptions
from modin.pandas import DataFrame

__SUPPORT_FILE_FORMAT__ = {
    "csv": "csv",
    "parquet": "parquet",
    "json": "json",
    "npy": "npy",
}


class FSRayDataSets:
    def __init__(self, label_col_name, data_source):
        """
        Args:
            label_col_name: 标签名
        """
        # 标签名
        self.label_col_name = label_col_name
        self.data_source = [data_source] if isinstance(data_source, str) else data_source
        self.file_format = self.__check_data_source_format(self.data_source)
        # 所有特征列表
        self.schemas = self._get_schemas()
        # read_data 中保存的 ref
        self.ds = None

    def read_data(self, *, data_source: Union[str, List[str]] = None,
                  filesystem: Optional["pyarrow.fs.FileSystem"] = None,
                  columns: Optional[List[str]] = None, parallelism: int = 200,
                  _tensor_column_schema: Optional[Dict[str, Tuple[np.dtype, Tuple[int, ...]]]] = None,
                  transformer: str = None, keep_ref=False,
                  **read_args):
        if data_source is None:
            # 如果是 读取全部数据
            if self.ds is not None:
                return self.ds
            data_source = self.data_source
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
        if transformer is None:
            # 仅当读取全部数据时保存一份
            if keep_ref:
                self.ds = ds
                return self.ds
            else:
                return ds
        elif transformer == "modin":
            if keep_ref:
                self.ds = ds.to_modin()
                return self.ds
            else:
                ds = ds.to_modin()
                return ds

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
            raise ValueError("data_source 格式错误！")

    """def get_column_by_name(self, column):
        print("进程：{} 正在获取列：{} 的数据".format(os.getpid(), column))
        if self.ds is not None and isinstance(self.ds, DataFrame):
            return self.ds[column].values"""

    def get_column_by_name_from_source(self, column: str, transformer: str = None):
        print("进程：{} 正在获取列：{} 的数据".format(os.getpid(), column))
        ds = self.read_data(columns=[column], transformer=transformer)
        if isinstance(ds, DataFrame):
            return ds[column].values
        else:
            return ds

    def _get_schemas(self):
        ds = self.read_data(data_source=self.data_source[0], keep_ref=False)
        schemas = ds.schema().names
        # 移除标签列
        schemas.remove(self.label_col_name)
        return schemas


if __name__ == '__main__':
    multi_path = ["D:\data\classify\giveMeSomeCredit\cs-training-no-index-part1.csv",
                  "D:\data\classify\giveMeSomeCredit\cs-training-no-index-part2.csv"]
    label1 = "SeriousDlqin2yrs"
    single_file_path = ["../data/final_test.csv"]
    label2 = "Response"
    dataset = FSRayDataSets(label1, multi_path)
    schemas_ = dataset.schemas
    print("schemas:{}".format(schemas_))
    print("schemas:{}".format(schemas_))
    # 按列获取数据
    for s in schemas_:
        s_d = dataset.get_column_by_name_from_source(s, "modin")
        print("特征：{} shape:{}".format(s, s_d.shape))

    whole_d = dataset.read_data(transformer="modin", keep_ref=True)
    print(whole_d.shape)
    print("-" * 20)
    print(dataset)
