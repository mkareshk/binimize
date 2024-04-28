import numpy as np
import pandas as pd
import pytest
import torch

from binimize import ExactSolver, QuboNumpy, QuboTorch


def test_exact_solver_with_numpy():
    Q = np.array([[1, 2], [2, 1]])
    qubo = QuboNumpy(Q)
    solver = ExactSolver(n_jobs=2)
    result = solver.solve(qubo)

    # Expected binary combinations and their QUBO values
    expected_results = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 6}

    # Verify DataFrame structure
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["x_1", "x_2", "value"]
    assert result.shape == (4, 3)  # 4 combinations, 2 variables + 1 value column

    # Verify all computed values match expected results
    for index, row in result.iterrows():
        x_tuple = tuple(row[:2].astype(int))
        assert row["value"] == expected_results[x_tuple]


# Tests for ExactSolver with QuboTorch
@pytest.mark.skipif(
    True or not torch.cuda.is_available(), reason="CUDA is not available"
)
def test_exact_solver_with_torch_cuda():
    Q = torch.tensor([[1, 2], [2, 1]], dtype=torch.float32)
    qubo = QuboTorch(Q, device="cuda")
    solver = ExactSolver(n_jobs=2)
    result = solver.solve(qubo)

    # Expected binary combinations and their QUBO values
    expected_results = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 6}

    # Verify DataFrame structure
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["x_1", "x_2", "value"]
    assert result.shape == (4, 3)  # 4 combinations, 2 variables + 1 value column

    # Verify all computed values match expected results
    for index, row in result.iterrows():
        x_tuple = tuple(row[:2].astype(int))
        assert row["value"] == expected_results[x_tuple]


# Run the tests
if __name__ == "__main__":
    pytest.main()
