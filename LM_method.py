import numpy as np
import matplotlib.pyplot as plt
from universal_function import plfunc_3D,armijo_wolfe
np.set_printoptions (suppress= True)


def lm_method(x0, f, g, H, max_iter=100, tol=1e-3, mu=1e-4):
    x = np.array([x0])#可以适用于任何函数
    gk = []

    for k in range(max_iter):
        gg = g(x[-1])
        Hessian = H(x[-1])
        I = np.eye(len(x0))

        # Levenberg-Marquardt modification
        LM_matrix = Hessian + mu * I
        d = -np.linalg.inv(LM_matrix) @ gg.T
        alpha = armijo_wolfe(x[-1], d, f, g)
        print(alpha)
        xk = x[-1] + alpha * d
        x = np.r_[x, [xk]]
        gk.append(np.linalg.norm(gg))

        # Update mu
        mu *= 2. if np.linalg.norm(d) < tol else 0.5
        #用方向的大小去反应Hessian的情况,若方向很小,说明矩阵不可逆,那么就修正,否则不修正(2*0.5=1)

        # Check convergence
        if np.linalg.norm(gg) < tol:
            print("Converged successfully.")
            break

    else:
        print("Exceeded maximum iterations.")

    return x, gk

def func(x):
    return 3 * x[0]**2 + 3 * x[1]**2 - (x[0]**2) * x[1]

def g(x):
    return np.array([6 * x[0] - 2 * x[0] * x[1], 6 * x[1] - x[0]**2])

def H(x):
    return np.array([[6 - 2 * x[1], -2 * x[0]], [-2 * x[0], 6]])

x0 = np.array([0,3])
result, gk = lm_method(x0, func, g, H)

print(result)
print(len(result))
print(gk)
xx = result.T[0]
yy = result.T[1]

# plot
plfunc_3D(func, result, [-6, 6], [-6, 6])