import math
import numpy as np
from typing import Callable, Tuple


def richardson(integrate: Callable[[int], float], h: float, eps: float) -> Tuple[float, float, float]:
    r = 1
    accuracy = math.inf
    opt_n = 0
    opt_step = 0
    
    while eps < accuracy:
        h_s = [h/2**i for i in range(r+2)]
        q_sums = [integrate(i) for i in range(1, r+3)]
        # print(q_sums)
        m = np.log(np.abs((q_sums[-1]-q_sums[-2])/(q_sums[-2]-q_sums[-3]))) / np.log(1/2)
        
        A = np.zeros((r+1, r+1), dtype=np.float64)
        A += [-1]
        for i in range(r+1):
            for j in range(1, r+1):
                A[i][j] = h_s[i] ** (m + j - 1)
        B = np.array(q_sums[:r+1], dtype=np.float64)
        x = np.linalg.solve(A, -B)
        J = x[0]
        
        print(q_sums[r], J) #!!!!!!!!!
        
        cur_eps = np.abs(q_sums[r] - J)
        if cur_eps < accuracy:
            accuracy = cur_eps
            opt_n = h / h_s[r-1]
            opt_step = h_s[r-1]       
        
        r += 1

    return accuracy, opt_n, opt_step
