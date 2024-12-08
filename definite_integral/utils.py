from scipy.integrate import quad

def accuracy(val, f, a, b):
    """Definite integral absolute accuracy"""
    integral = quad(f, a, b)
    return abs(integral[0] - val)

