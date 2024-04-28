import pytest
import torch

from binimize import QuboTorch


def test_valid_qubo_initialization_cpu():
    Q = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
    qubo = QuboTorch(Q, device="cpu")
    assert torch.equal(qubo.Q, Q.to("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_valid_qubo_initialization_gpu():
    Q = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
    qubo = QuboTorch(Q, device="cuda")
    assert torch.equal(qubo.Q, Q.to("cuda"))


def test_invalid_qubo_initialization():
    Q = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    with pytest.raises(ValueError):
        _ = QuboTorch(Q)


def test_qubo_with_valid_matrix():
    Q = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    qubo = QuboTorch(Q)
    assert torch.allclose(qubo.Q, Q)


def test_qubo_with_invalid_matrix_non_symmetric():
    Q = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        QuboTorch(Q)


def test_qubo_with_invalid_matrix_non_square():
    Q = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(ValueError):
        QuboTorch(Q)


def test_objective_value_cpu():
    Q = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
    qubo = QuboTorch(Q, device="cpu")
    x = torch.tensor([1, 0], dtype=torch.float32)
    assert qubo.objective_value(x) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_objective_value_gpu():
    Q = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
    qubo = QuboTorch(Q, device="cuda")
    x = torch.tensor([1, 0], dtype=torch.float32)
    assert qubo.objective_value(x) == 1


def test_objective_value_error_cpu():
    Q = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
    qubo = QuboTorch(Q, device="cpu")
    x = torch.tensor([1, 0, 1], dtype=torch.float32)
    with pytest.raises(ValueError):
        qubo.objective_value(x)


def test_var_num():
    Q = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
    qubo = QuboTorch(Q, device="cpu")
    assert qubo.var_num == 2


# Run tests
if __name__ == "__main__":
    pytest.main()
