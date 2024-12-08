import numpy as np


def count_moments(a, alpha: float, z_i0, z_i1: float) -> list[float]:
    mu0: float = (np.pow(z_i1-a, 1-alpha) - np.pow(z_i0-a,1-alpha)) / (1-alpha)
    mu1: float = (np.pow(z_i1-a, 2-alpha) - np.pow(z_i0-a, 2-alpha)) / (2-alpha) + a*mu0
    mu2: float = (np.pow(z_i1-a, 3-alpha) - np.pow(z_i0-a, 3-alpha)) / (3-alpha) + 2*a*mu1 - a*a*mu0
    
    return [mu0, mu1, mu2]


def count_weights(z_i1, z_i12, z_i2: float, mu: list[float]) -> list[float]:
    a1: float = (mu[2] - mu[1]*(z_i12 + z_i2) + mu[0]*z_i12*z_i2) / ((z_i12-z_i1)*(z_i2-z_i1))
    a2: float = (mu[2] - mu[1]*(z_i1+z_i2) + mu[0]*z_i1*z_i2) / ((z_i12-z_i1)*(z_i2-z_i12))
    a2 = -a2
    a3: float = (mu[2] - mu[1]*(z_i12+z_i1) + mu[0]*z_i12*z_i1) / ((z_i2-z_i12)*(z_i2-z_i1))

    return [a1, a2, a3]


def newton_kotes_f(f: callable, z_i0, z_i1: float, a, alpha: float) -> float:
    z_i01: float = (z_i0 + z_i1) / 2
    mu: list[float] = count_moments(a, alpha, z_i0, z_i1)
    a: float = count_weights(z_i0, z_i01, z_i1, mu)
    
    return a[0]*f(z_i0) + a[1]*f(z_i01) + a[2]*f(z_i1)


def c_newton_kotes_m(f: callable, a, b: float, alpha: float, k: int) -> float:
    step: float = (b-a)/k
    result: float = 0
    
    for i in range(1, k+1):
        z_i0, z_i1 = a+(i-1)*step, a+i*step
        result += newton_kotes_f(f, z_i0, z_i1, a, alpha)
        
    return result
