import numpy as np
from simplex import simplex

def test_simple_problem():
    c = np.array([3, 5])
    A = np.array([
        [1, 0],
        [0, 2],
        [3, 2]
    ])
    b = np.array([4, 12, 18])
    optimal_value, solution = simplex(c, A, b)
    assert np.isclose(optimal_value, 36)
    assert np.allclose(solution, [2, 6])

def test_unbounded_problem():
    c = np.array([1, 1])
    A = np.array([
        [-1, 1],
        [-1, -1]
    ])
    b = np.array([1, -2])
    optimal_value, solution = simplex(c, A, b)
    assert optimal_value == "unbounded"

def test_infeasible_problem():
    c = np.array([1, 1])
    A = np.array([
        [1, 1],
        [-1, -1]
    ])
    b = np.array([1, -2])
    optimal_value, solution = simplex(c, A, b)
    assert optimal_value == "infeasible"
