import torch


class QuboTorch:

    def __init__(self, Q, device="cpu"):
        self.device = torch.device(
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        Q = Q.to(self.device)
        if not self._is_valid_Q(Q):
            raise ValueError("Provided matrix Q is not valid for a QUBO problem.")
        self.Q = Q
        self.device = Q.device

    @staticmethod
    def _is_valid_Q(Q):
        """Check if the matrix Q is a valid QUBO matrix."""
        return (
            isinstance(Q, torch.Tensor)
            and Q.ndim == 2
            and Q.shape[0] == Q.shape[1]
            and torch.allclose(Q, Q.T)
        )

    def objective_value(self, x):
        """Calculate the objective value for a given binary vector x."""
        x = x.to(self.device)  # Ensure x is on the same device as Q
        if len(x) != self.Q.shape[0]:
            raise ValueError("Binary vector x and Q must have the same size.")
        return torch.dot(x, torch.matmul(self.Q, x))

    @property
    def var_num(self):
        """Returns the number of variables in the QUBO matrix."""
        return self.Q.size(0)