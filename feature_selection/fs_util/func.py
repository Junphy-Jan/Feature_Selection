import enum
import time
from functools import wraps

import numpy as np


class WrapTestFunc(object):
    def __init__(self, purpose=None, expect=None, timeit=True):
        self.purpose = purpose
        self.expect = expect
        self.timeit = timeit

    def __call__(self, func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            print("\n" + "-" * 15 + "运行：{} \t||\t测试内容：{}".format(func.__name__, self.purpose) +
                  "\t||\t期望:{}".format(self.expect) + "-" * 30)
            start = time.time()
            ret = func(*args, **kwargs)
            if self.timeit:
                print("-" * 15 + "函数：{} \t完成时间：{}".format(func.__name__, time.time() - start) + "-" * 15)
            return ret

        return wrapped_function


class SupportMethods(enum.Enum):
    VARIANCE = enum.auto()  # "低方差过滤"
    PEARSON_CORRELATION = enum.auto()  # "皮尔逊相关系数"
    CHI2 = enum.auto()  # "卡方检测"
    IV = enum.auto()  # "IV值过滤"
    NULL_FILTER = enum.auto()  # "空值过滤"
    PERMUTATION_IMPORTANCE = enum.auto()  # 特征置换重要性
    # IMPURITY_IMPORTANCE = enum.auto()  # 特征不纯度
    MUTUAL_INFO = enum.auto()  # 互信息
    FEATURE_IMPORTANCE = enum.auto()  # 嵌入式基于树模型的特征重要性排序
    STEP_WISE = enum.auto()  # 包裹式 stepwise 特征重要性


def is_continuous(feature, continuous_num=10):
    """
    判断 ndarray 是否是连续型数据，
    不同值数量 > 10 或 不同值数量大于数组长度 1/2 则认为是连续型

    Args:
        feature: 1d ndarray
        continuous_num: 连续型数据最小数据类别 unique > 10
    """
    if not np.issubdtype(feature.dtype, np.number):
        return False

    n = len(np.unique(feature))
    return n > continuous_num or n / len(feature) > 0.5


if __name__ == '__main__':
    continuous_array = np.arange(1, 10, dtype=np.float_)
    category_array = np.zeros((10,))
    print(continuous_array)
    print(category_array)
    assert True is is_continuous(continuous_array)
    assert False is is_continuous(category_array)
    m = [v.name for v in SupportMethods]
    print(m)
