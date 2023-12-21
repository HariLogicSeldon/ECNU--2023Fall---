import matplotlib.pyplot as plt
import autograd.numpy as np
def plfunc_3D(f,xlist,xrange,yrange):
    x1 = np.linspace(xrange[0], xrange[1], 100)#xrange,yrange是一个二元 numpyarray 或者 list
    x2 = np.linspace(yrange[0], yrange[1], 100)
    X, Y = np.meshgrid(x1, x2)
    Z = f((X, Y))#f是一个返回数字的函数
    fig, ax2 = plt.subplots()
    xx, yy = xlist[:, 0], xlist[:, 1]#xlist是一个 n 行两列的包含了路径数据的数组
    levels = np.linspace(np.min(Z), np.max(Z), 20)
    ax2.contour(X, Y, Z, levels=levels)
    ax2.scatter(xx, yy, c='r')
    ax2.plot(xx, yy)
    plt.show()

def initial_interval(alpha, f, x, d, gamma=0.08, t=0.9):
    i=0
    while True:
        new_alpha = alpha + gamma

        if new_alpha <= 0:
            new_alpha = 0
            if i == 0:
                gamma *= -1
                previous_alpha = new_alpha
            else:
                a, b = min(previous_alpha, new_alpha), max(previous_alpha, new_alpha)
                return a, b
        elif f(x + new_alpha * d) >= f(x + alpha * d):
            if i == 0:
                gamma *= -1
                previous_alpha = new_alpha
            else:
                a, b = min(previous_alpha, new_alpha), max(previous_alpha, new_alpha)
                return a, b
        else:
            gamma *= t
            previous_alpha = alpha
            alpha = new_alpha
            i += 1


def Gold_Step(f, x, d, alpha, epsilon=1e-3):
    a, b=0.8, alpha
    # print(a,b)
    r = 0.618034  # 黄金分割率

    while b - a > epsilon:
        a1, a2 = a + (1 - r) * (b - a), a + r * (b - a)

        if f(x + a1 * d) > f(x + a2 * d):
            a, a1, a2 = a1, a2, a + r * (b - a)
        else:
            b, a2, a1 = a2, a1, a + (1 - r) * (b - a)

    return (a + b) / 2

def armijo_wolfe(x, d, f, g, alpha=1, rho=1e-3, sigma=0.8):
    f0 = f(x)
    g0 = g(x)
    m = np.dot(g0, d)
    alpha_max = 1e10

    while alpha > 1e-10:
        x_new = x + alpha * d
        f_new = f(x_new)
        if f_new > f0 + rho * alpha * m or (f_new >= f(x) and alpha < 1e-10):
            alpha_max = alpha
            alpha = 0.6*alpha
            # print(alpha)
        else:
            g_new = g(x_new)
            if np.dot(g_new, d) < sigma * m or np.dot(g_new, d) > -sigma * m:
                alpha_max = alpha
                alpha = 0.6*alpha
                # print(alpha)
            elif np.dot(g_new, d) >= 0:
                alpha_max = alpha
                alpha = 0.6*alpha

            else:
                return alpha

    return alpha_max

def armijo(f,g,x,alpha,d,rho=1e-3):
    if f(x+alpha*d) <= f(x) + rho*g(x)*d*alpha:
        return True
    else:
        return False

