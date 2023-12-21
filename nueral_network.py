import autograd.numpy as np
from autograd import grad,elementwise_grad
from SR1 import sr1_method
import numpy as npp
from DFP import dfp_method
from BFGS import bfgs_method
import matplotlib.pyplot as plt

# 定义函数
def func(y):
    n=2
    v, theta, w = y[:n], y[n:2*n], y[2*n:3*n]
    m = 2
    x = np.linspace(0, 1, 3)
    value = 0

    for i in range(m):
        inner_value = 0
        for j in range(len(v)):
            expp = np.exp(theta[j] - w[j] * x[i])
            A = (1 + x[i]) * (v[j] / (1 + expp))
            B = (x[i] * v[j] * w[j] * expp / (1 + expp) ** 2)
            inner_value = inner_value + A + B

        value =  value + (-x[i] + inner_value) ** 2

    return value

# 使用 Autograd 计算梯度
grad = grad(func)


x0 = np.array([1.,1.,1.,1.,1,1.],dtype= float)
# print('SR1')
# result, gk = sr1_method(x0, func, grad)
# print(result[-1])
# print(gk)

# print('DFP')
# resultd, gkd = dfp_method(x0, func, grad)
# print(resultd[-1])
# # print(gkd)
#
# #
print('BFGS')
resultdb, gkdb = bfgs_method(x0, func, grad)
print(resultdb[-1])

# # print(gkdb)

def experiment_sol(x,para):
    v, theta, w = para[:2], para[2:4], para[4:6]
    m = 2
    value = 0.
    print(v,theta,w)
    for i in range(m):
        sig = 1./(1.+npp.exp(theta[i] - w[i] * x))
        value += v[i]*sig

    return x*value+1



def stander_sol(x):
    return x + npp.exp(-x)

X = npp.linspace(0,1,100)
Y = experiment_sol(X,resultdb[-1])
Y1 = stander_sol(X)
#
plt.plot(X,Y)
plt.plot(X,Y1)
plt.show()