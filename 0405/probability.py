# matplotlib inline
import torch
from torch.distributions import multinomial
import matplotlib.pyplot as plt
import numpy

fair_probs = torch.ones([6]) / 6
print(fair_probs)
a=multinomial.Multinomial(1, fair_probs).sample()
print(a)
b=multinomial.Multinomial(10, fair_probs).sample()
print(b)
# 将结果存储为32位浮点数以进行除法
counts = multinomial.Multinomial(1000, fair_probs).sample()
counts / 1000  # 相对频率作为估计值
print(counts / 1000)

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
print(counts.shape)
cum_counts = counts.cumsum(dim=0)
print(cum_counts)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
print(estimates.shape)

def set_figsize(figsize=(6, 4.5)):
    plt.figure(figsize=figsize)
set_figsize((6, 4.5))
for i in range(6):
    plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
plt.axhline(y=0.167, color='black', linestyle='dashed') #axhline()函数用于绘制水平参考线
plt.gca().set_xlabel('Groups of experiments')#gca get current axis
plt.gca().set_ylabel('Estimated probability')
plt.legend()
plt.show()
