import matplotlib.pyplot as plt
import numpy as np

from utils import accuracy
from data.test import a_coef, b_coef, f, alpha_coef, eps_coef

from method.formulas import *
from method.gauss import c_gauss_m
from method.newton_kotes import c_newton_kotes_m
from method.richardson import richardson


def task_1_1() -> None:
    # Task 1.1
    nodes = np.arange(10, 21, 1)
    diff = [0] * len(nodes)

    # average rectangle
    for i, n in enumerate(nodes):
        val = mid_rectangle_f(f, a_coef, b_coef, n)
        diff[i] = accuracy(val, f, a_coef, b_coef)
    plt.plot(nodes, diff, label="mid rectangle")

    # left rectangle
    for i, n in enumerate(nodes):
        val = left_rectangle_f(f, a_coef, b_coef, n)
        diff[i] = accuracy(val, f, a_coef, b_coef)
    plt.plot(nodes, diff, label="left rectangle")

    # trapeze
    for i, n in enumerate(nodes):
        val = trapeze_f(f, a_coef, b_coef, n)
        diff[i] = accuracy(val, f, a_coef, b_coef)
    plt.plot(nodes, diff, label="trapeze")

    # simpson
    for i, n in enumerate(nodes):
        val = simpson_f(f, a_coef, b_coef, n)
        diff[i] = accuracy(val, f, a_coef, b_coef)
    plt.plot(nodes, diff, label="simpson")

    plt.legend()
    plt.show()


def task_1_2() -> None:
    F = lambda x: f(x)/((x-a_coef)**alpha_coef)
    
    # Task 1.3
    nodes = np.arange(16, 100, 1)
    diff = [0] * len(nodes)
    
    # gauss
    for i, n in enumerate(nodes):
        val = c_gauss_m(f, a_coef, b_coef, alpha_coef, n)
        diff[i] = accuracy(val, F, a_coef, b_coef)
    plt.plot(nodes, diff, label="Gauss quadrature formula")
    
    # newton-kotes
    for i, n in enumerate(nodes):
        val = c_newton_kotes_m(f, a_coef, b_coef, alpha_coef, n)
        diff[i] = accuracy(val, F, a_coef, b_coef)
    plt.plot(nodes, diff, label="Newton-Kotes quadrature formula")

    plt.legend()
    plt.show()
    
    
def task_2_1() -> None:
    integrate = lambda n: c_gauss_m(f, a_coef, b_coef, alpha_coef, n)
    h = b_coef - a_coef
    print(
        richardson(integrate, h,  1e-9)
    )
    # integrate = lambda n: c_newton_kotes_m(f, a_coef, b_coef, alpha_coef, n)
    # print(
    #     richardson(integrate, h,  eps_coef)
    # )
