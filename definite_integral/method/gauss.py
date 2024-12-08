import numpy as np
from scipy.integrate import quad


# tested: works well
# $pytest - for testing
def cardano_f(p: list[float]) -> list[np.float64]:
    a, b, c, d = p[3], p[2], p[1], p[0]     # p=[2, -11, 12, 9] => 9x^3+12x^2-11x+2=0
    
    q = (2*b**3) / (54*a**3) - (b*c) / (6*a**2) + d/a/2
    p = (3*a*c-b**2) / (9*a**2)
    
    D = q**2 + p**3
    if D >= 0:
        raise ValueError('Equation has complex roots')
    
    r = np.sign(q)*np.sqrt(np.abs(p))
    phi = np.arccos(q/r**3)

    y1 = -2*r*np.cos(phi/3) - b/(3*a)
    y2 = 2*r*np.cos(np.pi/3 - phi/3) - b/(3*a)
    y3 = 2*r*np.cos(np.pi/3 + phi/3) - b/(3*a)

    return [y1, y2, y3]


# if nothing works, check it out more carefully
def count_weights(z0, z1: float, alpha: float, n: int) -> list[np.float64]:
    """Counting weights for gauss quadrature formula.
    Weight is integral by f(t)=t^(j-alpha) on [z0, z1] where j belongs to [0, 2*n-1].
    """
    weights = np.array([0]*(n), dtype=np.float64)
    for j in range(n):
        weights[j] = z1**(j-alpha+1)/(j-alpha+1) - z0**(j-alpha+1)/(j-alpha+1)
        
    return weights


def gauss_f(f_: callable, z0, z1: float, alpha: float, n: int) -> float:
    # step 1
    # counting weights
    mu = count_weights(z0, z1, alpha, 2*n)   
    
    # step 2
    # solving SLAE of type Ax=B to find out a_j
    A = np.zeros((n, n), dtype=np.float64)
    B = np.zeros(n, dtype=np.float64)
    for s in range(n):
        for j in range(n):
            A[s][j] = mu[s+j]
        B[s] = -mu[s+3]

    a_j = np.linalg.solve(A, B)

    # step 3
    # solving cubic euqation with cardano formula
    a_j = np.append(a_j, 1)
    x_j = cardano_f(a_j)
    
    # step 4 
    # solving SLAE of type Ax=B to find out quadrature coefficients
    A = np.zeros((n, n), dtype=np.float64)
    B = np.zeros(n, dtype=np.float64)
    for s in range(n):
        for j in range(n):
            A[s][j] = np.pow(x_j[j], s)
        B[s]= mu[s]
    A_j = np.linalg.solve(A, B)
    
    # final
    res = f_(x_j[0])*A_j[0] + f_(x_j[1])*A_j[1] + f_(x_j[2])*A_j[2]
    return res


def c_gauss_m(f: callable, a, b: float, alpha: float, k: int) -> float:
    n = 3
    shift = a
    f_: callable = lambda t: f(t+shift)
    a, b = 0, b-a
    
    step: float = (b-a)/k
    res: float = 0

    for i in range(1, k+1):
        z0, z1 = a+(i-1)*step, a+i*step
        res += gauss_f(f_, z0, z1, alpha, n)
    
    return res
