import numpy as np
import matplotlib.pyplot as plt

import torch as th
# x = np.linspace(-2*np.pi, 2*np.pi, 100)
# y = np.sin(x)

# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('sin(x)')
# plt.title('Plot of sin(x)')
# plt.grid(True)
# plt.show()

x=th.arange(0,10,0.1)
print(x)

y=th.arange(12)
print(y)
print(y.shape)
print(y.numel())
y=y.reshape(3,4)
print(y)

z=th.zeros(5,2,2)
print(z)

sigma=th.ones(3,2,4)
print(sigma)

beita=th.randn(3,2,4)
print(beita)


x=th.tensor([1,2,3,4,5])
y=th.tensor([2,2,3,2,2])
print(x+y)
print(x**y)
z=th.exp(x)
print(z)

x=th.arange(12,dtype=th.float32).reshape(3,4)
y=th.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])

print(th.cat((x,y),dim=0))
print(th.cat((x,y),dim=1))
z=x==y
print(z)
print(x.sum())
print(x[-1])
print(y[1:3])
print(x[-1,2])
print(x[-1,0])
a=x.numpy()
b=th.tensor(a)
print(type(a),type(b))
z=x<y
print(z)
z=x>y
print(z)

