import autograd.numpy as npa
from autograd import grad
from SR1 import *
# 定义函数
def your_function(y):
    v, theta, w = y[:2], y[2:4], y[4:6]
    m = 3
    x = npa.linspace(0, 1, m)
    value = 0

    for i in range(m):
        inner_value = 0
        for j in range(len(v)):
            exp = npa.exp(theta[j] - w[j] * x[i])
            inner_value += (1 + x[i]) * (v[j] / (1 + exp)) + (x[i] * v[j] * w[j] * exp) / (1 + exp) ** 2

        value += (-x[i] + inner_value) ** 2

    return value

# 使用 Autograd 计算梯度
grad_your_function = grad(your_function)

result, gk = sr1_method(x0, your_function, grad_your_function)