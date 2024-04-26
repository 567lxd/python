# activation function
import sys
sys.path.append("/Users/xiaodongliang/learning/python")
import self_package.torch as d2l
import torch
from torch import nn


x=torch.arange(-8.0,8.0,0.1,requires_grad=True)
y=torch.relu(x)
# d2l.plot(x.detach(),y.detach(),'x','relu(x)',figsize=(5,2.5))

y.backward(torch.ones_like(x),retain_graph=True)
# d2l.plot(x.detach(),x.grad,'x','grad of relu',figsize=(5,2.5))

y=torch.sigmoid(x)
# d2l.plot(x.detach(),y.detach(),'x','sigmoid(x)',figsize=(5,2.5))

#清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
# d2l.plot(x.detach(),x.grad,'x','grad of sigmoid',figsize=(5,2.5))

y=torch.tanh(x)
# d2l.plot(x.detach(),y.detach(),'x','tanh(x)',figsize=(5,2.5))

x.grad.data.zero_() 
y.backward(torch.ones_like(x),retain_graph=True)
# d2l.plot(x.detach(),x.grad,'x','grad of tanh',figsize=(5,2.5))


batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

num_inputs,num_outputs,num_hiddens=784,10,256

w1=nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True)*0.01)
b1=nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))

w2=nn.Parameter(torch.randn(num_hiddens,num_outputs,requires_grad=True)*0.01)
b2=nn.Parameter(torch.zeros(num_outputs,requires_grad=True))

params=[w1,b1,w2,b2]

# activation function
def relu(x):
    a=torch.zeros_like(x)
    return torch.max(x,a)

# model
def net(x):
    x=x.reshape((-1,num_inputs))
    H=relu(x@w1+b1)#@矩阵乘法
    return (H@w2+b2)

#cost function
loss=nn.CrossEntropyLoss(reduction='none')

#train
num_epochs,lr=10,0.1
updater=torch.optim.SGD(params,lr=lr)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)
d2l.predict_ch3(net,test_iter)
d2l.plt.show()


    