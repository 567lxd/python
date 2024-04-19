import sys
# import os
# print(os.path.exists("/Users/xiaodongliang/learning/python/self_package"))
sys.path.append("/Users/xiaodongliang/learning/python")
# print(sys.path)
import self_package.self_plot as self_plot
import numpy as np 
import matplotlib_inline.backend_inline as backend_inline
import matplotlib.pyplot as plt
import torch

def f(x):
    return 3*x**2-4*x

def numerical_lim(f,x,h):
    return (f(x+h)-f(x))/h

h=0.1
for i in range(5):
    print(f'h={h},numerical limit={numerical_lim(f,1,h)}')
    h*=0.1

x = np.arange(0, 3, 0.1)
self_plot.plot(x, [f(x), 2 * x - 3,x+1], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
