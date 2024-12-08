"""Quadrature formulas for task 1.1"""


def mid_rectangle_f(f, a, b, n):
    """Mid rectangle quadrature formula"""
    h = (b - a) / (n - 1)
    x = [(a + i * h) for i in range(0, n)]

    result = 0
    for i in range(len(x) - 1):
        middle = (x[i] + x[i + 1]) / 2
        result += h * f(middle)

    return result


def left_rectangle_f(f, a, b, n):
    """Left rectangle quadrature formula """
    h = (b - a) / (n - 1)
    x = [(a + i * h) for i in range(0, n)]

    result = 0
    for i in range(len(x) - 1):
        result += h * f(x[i])

    return result


def trapeze_f(f, a, b, n):
    """Trapezoid quadrature formula"""
    h = (b - a) / (n - 1)
    x = [(a + i * h) for i in range(0, n)]

    result = 0
    result += (h * (f(a) + f(b)) / 2)
    for i in range(1, len(x) - 1):
        result += h * f(x[i])

    return result


def simpson_f(f, a, b, nodes):
    """Simpson formula"""
    # interval count
    n = nodes - 1
    # step
    h = (b - a) / (2 * n)
    x = [(a + i * h) for i in range(2 * n + 1)]

    result = 0
    result += (f(x[0]) + f(x[-1]))

    for i in range(1, n):
        result += 4 * f(x[2 * i - 1])
        result += 2 * f(x[2 * i])
    result += 4 * f(x[2 * n - 1])
    result = result * h / 3

    return result
