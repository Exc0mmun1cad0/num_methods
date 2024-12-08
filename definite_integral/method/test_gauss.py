import numpy as np

from gauss import cardano_f


def test_cardano_f():
    test_cases = (
        [5, -8, -8, 5],
        [2, -11, 12, 9],
        [9, 12, 11, 2],
    )
    test_answers = (
        np.array([-1, 0.46933761370819, 2.1306623862918], dtype=np.float64),
        np.array([-2, 0.33333333, 0.33333333], dtype=np.float64),
        np.array([-0.5, 3.00000000, 3.00000000], dtype=np.float64),
    )
    
    for i, test_case in enumerate(test_cases):
        test_res = cardano_f(test_case)
        test_res = sorted(test_res)
        assert(all([test_answers[i][j] - test_res[j] < 1e-7 for j in range(3)]))

