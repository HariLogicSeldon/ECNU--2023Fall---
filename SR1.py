import autograd.numpy as np
import matplotlib.pyplot as plt
from universal_function import plfunc_3D,armijo_wolfe
np.set_printoptions(suppress=True)

def sr1_update(B, s, y):
    y = y - B @ s
    B += np.outer(y, y) / np.dot(y, s)
    return B

def sr1_method(x0, f, g, max_iter=100, tol=1e-3, mu=1e-4):
    x = np.array([x0],dtype=float)
    gk = []
    Hessian = np.eye(len(x0))  # 初始估计的 Hessian 矩阵为单位矩阵

    for k in range(max_iter):
        gg = g(x[-1])
        d = -np.linalg.inv(Hessian) @ gg.T
        alpha = armijo_wolfe(x[-1], d, f, g)
        s = alpha * d
        xk = x[-1] + s
        x = np.r_[x, [xk]]
        gk.append(np.linalg.norm(gg))

        y = g(xk) - gg
        Hessian = sr1_update(Hessian, s, y)

        # Check convergence
        if np.linalg.norm(gg) < tol:
            print("Converged successfully.")
            break

    else:
        print("Exceeded maximum iterations.")

    return x, gk

if __name__ =="__main__":
    def func(x):
        return 3 * x[0]**2 + 3 * x[1]**2 - (x[0]**2) * x[1]

    def g(x):
        return np.array([6 * x[0] - 2 * x[0] * x[1], 6 * x[1] - x[0]**2])

    x0 = np.array([-2,4])
    result, gk = sr1_method(x0, func, g)

    print(result)
    print(len(result))
    print(gk)
    xx = result.T[0]
    yy = result.T[1]

    # plot
    plfunc_3D(func, result, [-6, 6], [-6, 6])

