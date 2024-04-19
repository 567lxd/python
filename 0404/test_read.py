import pandas as pd
import numpy as np
import torch as th
data = pd.read_csv('data/data.csv')
print(data)
# 插值法
""" inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
numeric_columns = inputs.select_dtypes(include=['float64', 'int64']).columns
print(numeric_columns)
inputs[numeric_columns] = inputs[numeric_columns].fillna(
    inputs[numeric_columns].mean())
# inputs=inputs.fillna(inputs.mean())
print(inputs)
print(outputs)
outputs = outputs.fillna(outputs.mean())
intputs = pd.get_dummies(inputs, dummy_na=True)
print(intputs)
intputs = th.tensor(intputs.to_numpy(dtype=float))
outputs = th.tensor(outputs.to_numpy(dtype=float))
outputs = outputs.view(-1,1)
print(intputs)
print(outputs)
# conbindata = pd.concat([inputs, outputs], axis=1)
# print(conbindata)
print(th.cat((intputs, outputs), dim=1)) """
#插值法

#删除法
# inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
data.dropna(axis=1,inplace=True)
print(data)
#删除法



""" 选择插值法还是删除法取决于数据的具体情况以及分析的目的。
插值法可以保留更多的数据信息，并且不会改变数据的整体分布特征，但是可能会引入一定的误差。
删除法则可以简化数据集，减少对后续分析的影响，但是会减少数据量，可能会丢失一些重要信息。 """

