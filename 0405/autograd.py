import torch
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.pylab as plt
import torch as th

x = th.arange(4, requires_grad=True, dtype=th.float32)
x.grad
print(x.grad)
y = 2*th.dot(x, x)
y.backward()
x.grad
print(x.grad)
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
print(y)
y.backward()
x.grad
print(x.grad)

x.grad.zero_()
y = x*x
print(y)
print(y.sum())
y.sum().backward()
x.grad
print(x.grad)

x.grad.zero_()
y = x * x
# 在 PyTorch 中，detach() 方法用于创建一个新的张量，该张量从原始张量中分离出来，并且不会反向传播其梯度。这是一种常见的方法，用于阻止特定张量的梯度被计算，而不影响其他张量
u = y.detach()
z = u * x

z.sum().backward()
print(x.grad == u)
print(x.grad)
# z.sum().backward()
# print(x.grad == u)
# print(x.grad)

""" def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a=th.randn(size=(),requires_grad=True)
d=f(a)
d.backward()
print(a.grad == d / a)
print(a.grad)
 """


def f(a):
    b = a * 2
    while b.norm() < 1000:
        # print(“\n”,b.norm())
        b = b * 2
    if b.sum() > 0:
        c = b
        # print(“C==b\n”,c)
    else:
        c = 100 * b
        # print(“c=100b\n”,c)
    return c


a = th.randn(size=(3, 1), requires_grad=True)
print(a.shape)
print(a)
d = f(a)
# d.backward() #<====== run time error if a is vector or matrix RuntimeError: grad can be implicitly created only for scalar outputs
d.sum().backward()  # <===== this way it will work
print(d)


def f(a):
    b = a*a
    u = b.detach()
    c = a * u
    c.sum().backward()

a.grad.zero_()
a = th.randn(size=(3, 1), requires_grad=True)
print(a)
f(a)
print(a.grad)
print(a*a)
print(a.grad == a*a)


""" # matplotlib inline

f, ax = plt.subplots(1)

x = np.linspace(-3*np.pi, 3*np.pi, 100)
x1 = torch.tensor(x, requires_grad=True)
y1 = torch.sin(x1)
y1.sum().backward()

ax.plot(x, np.sin(x), label='sin(x)')
ax.plot(x, x1.grad, label='gradient of sin(x)')

ax.legend(loc='upper center', shadow=True)

ax.xaxis.set_major_formatter(FuncFormatter(
    lambda val, pos: '{:.0g}$\pi$'.format(val/np.pi) if val != 0 else '0'
))
ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))

plt.show() """
