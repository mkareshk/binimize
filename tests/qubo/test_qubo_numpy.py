import numpy as np
import pytest

from binimize import QuboNumpy


def test_valid_qubo_initialization():
    Q = np.array([[1, 2], [2, 3]])
    qubo = QuboNumpy(Q)
    assert np.array_equal(qubo.Q, Q)


def test_qubo_with_valid_matrix():
    Q = np.array([[1, 2], [2, 1]])
    qubo = QuboNumpy(Q)
    assert np.array_equal(qubo.Q, Q)


def test_qubo_with_invalid_matrix_non_symmetric():
    Q = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        QuboNumpy(Q)


def test_qubo_with_invalid_matrix_non_square():
    Q = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        QuboNumpy(Q)


def test_qubo_objective_value():
    Q = np.array([[1, 2], [2, 3]])
    qubo = QuboNumpy(Q)
    x = np.array([1, 0])
    assert qubo.objective_value(x) == 1


def test_qubo_objective_value_error():
    Q = np.array([[1, 2], [2, 3]])
    qubo = QuboNumpy(Q)
    x = np.array([1, 0, 1])
    with pytest.raises(ValueError):
        qubo.objective_value(x)


def test_var_num():
    Q = np.array([[1, 2], [2, 3]])
    qubo = QuboNumpy(Q)
    assert qubo.var_num == 2


# Run tests
if __name__ == "__main__":
    pytest.main()
