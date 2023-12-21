import autograd.numpy as anp
from autograd import grad
from universal_function import armijo_wolfe
import numpy as np
from scipy.sparse import lil_matrix

def prp_beta(g_old,g_new):
    up = np.dot(g_new,g_new-g_old)
    d = np.dot(g_old,g_old)
    return up/d

def prp_method(x0, f,  g, max_iter=1000, tol=1e-6):
    x = np.array([x0],dtype=float)
    g_norm = []
    g_list = []
    dlist = []

    g_ini = g(x[-1])
    g_list.append(g_ini)
    g_norm.append(np.linalg.norm(g_ini))
    d = -g_ini
    dlist.append(d)

    for k in range(max_iter):
        print(k)

        #更新 x
        alpha = armijo_wolfe(x[-1], dlist[-1], f, g)
        s = alpha * dlist[-1]
        xk = x[-1] + s
        x = np.r_[x, [xk]]

        g_new = g(xk)
        g_norm.append(np.linalg.norm(g_new))
        g_list.append(g_new)

        beta = prp_beta(g_list[-2],g_list[-1])

        # print("newd",-g_new + beta * dlist[-1])
        dlist.append(-g_new + beta * dlist[-1])

        # Check convergence
        if g_norm[-1] <= tol*g_norm[0]:
            print("Converged successfully.")
            print("Iterate Times",k)
            break

    else:
        print("Exceeded maximum iterations.")

    return x, g_norm

def BB_alpha(G,g_old):
    u = np.dot(g_old,G)
    u = np.dot(u,g_old)
    d = np.dot(g_old,G)
    d = np.dot(d,G)
    d = np.dot(d,g_old)

    return u/d

def BB_method(x0, G,  g, max_iter=1000, tol=1e-6):
    x = np.array([x0], dtype=float)
    g_norm = []
    g_list = []
    g_ini = g(x[-1])
    g_list.append(g_ini)
    g_norm.append(np.linalg.norm(g_ini))


    for k in range(max_iter):
        print(k)

        #更新 x
        if k ==0 :
            alpha = 1
        else:
            alpha = BB_alpha(G, g_list[-2])
        s = alpha * -g_list[-1]
        xk = x[-1] + s
        x = np.r_[x, [xk]]

        g_new = g(xk)
        g_norm.append(np.linalg.norm(g_new))
        g_list.append(g_new)


        # Check convergence
        if g_norm[-1] <= tol*g_norm[0]:
            print("Converged successfully.")
            print("Iterate Times",k)
            break

    else:
        print("Exceeded maximum iterations.")

    return x, g_norm




if __name__ == '__main__':

    def create_laplace_matrix(n):
        # n 表示每个维度上的网格点数量
        total_points = n ** 3
        laplace_matrix = lil_matrix((total_points, total_points))

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    idx = i * n ** 2 + j * n + k  # 当前点在一维数组中的索引

                    # 设置对角元素
                    laplace_matrix[idx, idx] = 6

                    # 设置相邻点的元素
                    if i > 0:
                        laplace_matrix[idx, idx - n ** 2] = -1
                    if i < n - 1:
                        laplace_matrix[idx, idx + n ** 2] = -1
                    if j > 0:
                        laplace_matrix[idx, idx - n] = -1
                    if j < n - 1:
                        laplace_matrix[idx, idx + n] = -1
                    if k > 0:
                        laplace_matrix[idx, idx - 1] = -1
                    if k < n - 1:
                        laplace_matrix[idx, idx + 1] = -1

        return laplace_matrix


    # 生成一个10x10x10的网格
    n = 20
    laplace_matrix = create_laplace_matrix(n)

    G = laplace_matrix.toarray()
    print(G)


    def X_star(u, v, w):
        A = u * (u - 1) * v * (v - 1) * w * (w - 1)
        sigma, alpha, beta, gamma = 20., 0.5, 0.5, 0.5
        B = anp.exp(-(pow(sigma, 2) * ((u - alpha) ** 2 + (v - beta) ** 2) + (w - gamma) ** 2) / 2)
        return A * B


    # 生成一些网格点
    u_values = np.linspace(0, 1, 20)
    v_values = np.linspace(0, 1, 20)
    w_values = np.linspace(0, 1, 20)
    Xstar = []
    # 在网格点上计算 X_star 的值
    for u in u_values:
        for v in v_values:
            for w in w_values:
                result = X_star(u, v, w)
                Xstar.append(result)

    Xstar = np.array(Xstar)

    b = np.dot(G, Xstar)


    def f(x):
        # 二次项
        quad = np.dot(x, G)
        quad = np.dot(quad, x)
        # 一次项
        one = np.dot(b, x)

        return 0.5 * quad - one


    def g(x):
        return np.dot(x, G) - b


    x0 = np.zeros(8000)

    x, gk = prp_method(x0, f, g)
    print(x)
    print(gk)

    x1,gk1 = BB_method(x0, G,  g, max_iter=1000, tol=1e-6)

    print(x1,gk1)





