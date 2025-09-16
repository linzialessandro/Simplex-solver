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
    constraint_types = ['<=', '<=', '<=']
    optimal_value, solution = simplex(c, A, b, constraint_types)
    assert np.isclose(optimal_value, 36)
    assert np.allclose(solution, [2, 6])

def test_unbounded_problem():
    c = np.array([1, 1])
    A = np.array([
        [-1, 1],
        [-1, -1]
    ])
    b = np.array([1, -2])
    constraint_types = ['<=', '<=']
    optimal_value, solution = simplex(c, A, b, constraint_types)
    assert optimal_value == "unbounded"

def test_greater_than_constraint():
    c = np.array([-3, -2])
    A = np.array([
        [2, 1],
        [-3, 2],
        [1, 1]
    ])
    b = np.array([10, 6, 6])
    constraint_types = ['>=', '<=', '>=']
    optimal_value, solution = simplex(c, A, b, constraint_types)
    assert np.isclose(optimal_value, -16)
    assert np.allclose(solution, [4, 2])


def test_equality_constraint():
    c = np.array([3, 2])
    A = np.array([
        [1, 1],
        [2, 1]
    ])
    b = np.array([7, 10])
    constraint_types = ['==', '<=']
    optimal_value, solution = simplex(c, A, b, constraint_types)
    assert np.isclose(optimal_value, 17)
    assert np.allclose(solution, [3, 4])


def test_negative_b_value():
    c = np.array([1, 1])
    A = np.array([
        [1, 1]
    ])
    b = np.array([-5])
    constraint_types = ['<=']
    optimal_value, solution = simplex(c, A, b, constraint_types)
    assert optimal_value == "infeasible"


def test_infeasible_problem():
    c = np.array([1, 1])
    A = np.array([
        [1, 1],
        [-1, -1]
    ])
    b = np.array([1, -2])
    constraint_types = ['<=', '<=']
    optimal_value, solution = simplex(c, A, b, constraint_types)
    assert optimal_value == "infeasible"
