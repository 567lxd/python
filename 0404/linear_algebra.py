import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th

A=th.arange(16).reshape(4,4)
print(A)
B=th.arange(20).reshape(4,5)
print(len(A))#tensor的长度
print(A.shape)#tensor的形状
print(A.size())#tensor的大小
print(A[0])#取第一行
print(A[:,0])#取第一列
print(A[0][0])#取第一个元素

print("A end")

print(B)
print(len(B))#tensor的长度
print(B.shape)#tensor的形状
print(B.size())#tensor的大小
print(B[0])#取第一行
print(B[:,0])#取第一列
print(B[0][0])#取第一个元素
print("B end")


print(A==A.T)#判断是否对称

A=th.arange(20,dtype=th.float32).reshape(4,5)
B=A.clone()#克隆一个A,分配新内存
C=A.clone().detach()#克隆一个A,不分配新内存，detach()方法用于创建一个与原始张量共享数据的新张量，但是不共享梯度信息。梯度信息是在深度学习中用于反向传播算法的一种重要信息。

print(A)
print(A+B)
print(id(A),id(B),id(C))

a=th.arange(24,dtype=th.float32).reshape(2,3,4)
print(a)
print(a+2)
print(a*2)#hadamard乘积
print((a*2).shape)


#降维
print(a.sum())#所有元素求和
print(a.sum(axis=0))#第一维求和
print(a.sum(axis=1))#第二维求和
print(a.sum(axis=2))#第三维求和

print(a.mean(axis=0))#求平均
print(a.sum(axis=0)/a.shape[0])#求平均
print(a.shape)
print(a)


x=th.arange(4,dtype=th.float32)
print(x)
y=th.ones(4,dtype=th.float32)
print(y)

print(th.dot(x,y))#点乘
print(th.sum(x*y))#点乘


A=th.arange(20).reshape(5,4)
B=th.arange(4)
print(A)
print(B)
print(th.mv(A,B))#矩阵乘法
print(th.mv(A,B).shape)

A=th.arange(20).reshape(5,4)
B=th.arange(20).reshape(4,5)
print(A)
print(B)
print(th.mm(A,B))#矩阵乘法
print(th.mm(A,B).shape)


C=th.ones(3,4)
print(C)