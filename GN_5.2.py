import numpy as np
import matplotlib.pyplot as plt
from universal_function import armijo_wolfe

def f(x,lam=0.1):
    return 0.5*((x+1)**2 + (lam * x**2 + x -1)**2)

def ad(x,lam):
    J = np.array([1,2*lam*x+1])
    JJ = np.dot(J,J)
    Jr = 2 * (lam**2) * (x**3) + 3 * lam * (x**2) - 2 * lam * x + 2 * x
    return Jr/JJ

def J(x,lam=0.1):
    return np.array([1.,2.*lam*x+1])
def r(x,lam=0.1):
    return np.array([x+1,lam * x**2 + x - 1])

def Gauss_Newton(x0,J,r,tol=1e-6,max_iter=100):
    x = np.array([x0], dtype=float)

    gk = []

    for k in range(max_iter):
        gg = J(x[-1])
        rr = r(x[-1])
        JJ = np.dot(gg,gg)
        Jr = np.dot(gg,rr)
        if len(gg.shape)==1:
            d = -(1/JJ) * np.dot(gg,rr)
        else:
            JJ.astype(float)
            d = -np.linalg.inv(JJ) * np.dot(gg,rr)

        alpha = 0.7
        s = alpha*d
        print(s)
        xk = x[-1] + s
        x = np.r_[x, [xk]]
        gk.append(np.linalg.norm(Jr))
        # Check convergence
        if np.linalg.norm(Jr) < tol:
            print("Converged successfully.")
            break

    else:
        print("Exceeded maximum iterations.")

    return x, gk

def householder_reflection(A):
    (m,n) = np.shape(A)
    Q = np.identity(r)
    R = np.copy(A)
    for cnt in range(r-1):
        x = R[cnt:,cnt]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x-e
        v = u/np.linalg.norm(u)
        Q_cnt = np.identity(r)
        Q_cnt[cnt:,cnt:] -= 2 * np.outer(v,v)
        R = np.dot(Q_cnt,R) #R = H(n-1)*...*H(2)*H(1)*A
        Q = np.dot(Q,Q_cnt)
        return Q,R

def QR_method(x,J,r):
    JJ = J(x)
    Q,R = householder_reflection(J)
    rr = r(x)
    b = np.dot(Q,r)
    R = R.astype(float)
    R_inv = np.linalg.inv(R)
    return np.dot(R_inv,-b)

if __name__ == "__main__":
    x0 = 0.5
    lam = 0.1
    xGN,gk = Gauss_Newton(x0, J, r)
    print(xGN)
    print(gk)

    lam = 0.1
    x = np.linspace(-1,1,100)
    y = f(x,lam)
    max_iter = 100
    xlist = [0.7]
    epsilon = 1e-5
    for k in range(max_iter):
        x_new = xlist[-1] -  ad(xlist[-1],lam)
        xlist.append(x_new)
    xlist = np.array(xlist)
    xlist = np.abs(xlist)
    Ypred = np.log10(xlist)
    yy = f(xlist,lam)
    yGN = f(xGN)

    klist = np.arange(101)
    print(len(xlist))
    f, (ax1, ax2) = plt.subplots(1, 2,)


    ax1.plot(klist,Ypred,c="r")

    ax2.plot(x, y)
    ax2.scatter(xGN,yGN,c=yGN,label="GS")
    ax2.legend()


    plt.show()
