from numpy import cos, sin, exp
import math


def f(x):
    """7th option of the function f() for the task"""
    return 4.5 * cos(7 * x) * exp((-2) * x / 3) + 1.4 * sin(1.5 * x) * exp((-x) / 3) + 3

def p(x):
    """7th option of the function p() for the task"""
    return (x - a_coef)**(-alpha_coef)

F = lambda x: p(x)*f(x)

# [a, b] - integration interval
a_coef = 2.1
b_coef = 3.3

# For task 1.2
alpha_coef = 0.4

# For tasks 2
eps_coef = 1e-6
