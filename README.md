# Feature_Selection
基于ray的特征选择库，集成多种方法 (a tool for feature selection based on ray distributed framework)

| 方法名                                           | 变量数据   | 限制                      | 排序取值 |
| ------------------------------------------------ | ---------- | ------------------------- | -------- |
| variance(低方差过滤)                             | 单变量     | 未归一化的数据            | 大       |
| chi-square(卡方值过滤)                           | 变量与标签 | 特征值非负，且不能nan/inf | 大       |
| Pearson-value(皮尔森系数)                        | 变量与标签 | 两个变量需服从正态分布    | 大       |
| IV-value(IV值过滤)                               | 变量与标签 | 值不能为nan/inf           | 大       |
| 空值率                                           | 单变量     |                           | 小       |
| 互信息(mutual info)                              | 变量与标签 | 值不能为nan/inf           | 大       |
| 特征置换重要性(permutation importance)           | 全部数据   | 需要已训练的模型          | 大       |
| 特征不纯度/树模型特征重要性(impurity importance) | 全部数据   |                           | 大       |
| 逐步回归(stepwise)                               | 全部数据   |                           | 小       |



以上方法在使用时可指定。

# 用法
参考 ```feature_selection\test_fs\test_func_flow.py```
