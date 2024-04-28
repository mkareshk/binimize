import numpy as np
import pytest
import torch

from binimize import QUBO, QuboNumpy, QuboTorch


def test_qubo_numpy_valid_initialization():
    Q = np.array([[1, 2], [2, 1]])
    qubo_instance = QUBO(Q, backend="numpy")
    assert isinstance(qubo_instance, QuboNumpy)
    assert qubo_instance.var_num == 2
    assert np.array_equal(qubo_instance.Q, Q)


def test_qubo_torch_valid_initialization():
    Q = torch.tensor([[1, 2], [2, 1]], dtype=torch.float32)
    qubo_instance = QUBO(Q, backend="torch")
    assert isinstance(qubo_instance, QuboTorch)
    assert qubo_instance.var_num == 2
    assert torch.equal(qubo_instance.Q, Q)


def test_qubo_invalid_backend():
    Q = np.array([[1, 2], [2, 1]])
    with pytest.raises(ValueError):
        _ = QUBO(Q, backend="invalid")


def test_qubo_numpy_objective_value():
    Q = np.array([[1, 2], [2, 1]])
    qubo_instance = QUBO(Q, backend="numpy")
    x = np.array([1, 0])
    assert qubo_instance.objective_value(x) == 1


def test_qubo_torch_objective_value():
    Q = torch.tensor([[1, 2], [2, 1]], dtype=torch.float32)
    qubo_instance = QUBO(Q, backend="torch")
    x = torch.tensor([1, 0], dtype=torch.float32)
    assert qubo_instance.objective_value(x) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_qubo_torch_on_gpu():
    Q = torch.tensor([[1, 2], [2, 1]], dtype=torch.float32)
    qubo_instance = QUBO(Q, backend="torch", device="cuda")
    assert qubo_instance.Q.device.type == "cuda"


def test_qubo_numpy_on_cpu():
    Q = np.array([[1, 2], [2, 1]])
    qubo_instance = QUBO(Q, backend="numpy")
    assert isinstance(qubo_instance.Q, np.ndarray)


# Run tests
if __name__ == "__main__":
    pytest.main()
