import numpy as np
import matplotlib.pyplot as plt
from universal_function import plfunc_3D,armijo_wolfe
np.set_printoptions(suppress=True)



def BFGS_update(B, s, y):
    Bs = B @ s
    return B + np.outer(y, y) / np.dot(y, s) - np.outer(Bs, Bs) / np.dot(s, Bs)

def bfgs_method(x0, f, g, max_iter=1000, tol=1e-2):
    x = np.array([x0],dtype=float)
    gk = []
    B = np.eye(len(x0))  # 初始估计的 Hessian 逆矩阵

    for k in range(max_iter):
        gg = g(x[-1])
        d = -np.linalg.inv(B) @ gg.T
        alpha = 1.116
        s = alpha * d
        xk = x[-1] + s
        x = np.r_[x, [xk]]
        gk.append(np.linalg.norm(gg))

        y = g(xk) - gg
        B = BFGS_update(B, s, y)

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
    result, gk = bfgs_method(x0, func, g)

    print(result)
    print(len(result))
    print(gk)
    xx = result.T[0]
    yy = result.T[1]

    # plot
    plfunc_3D(func, result, [-6, 6], [-6, 6])
