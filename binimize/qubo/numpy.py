import numpy as np


class QuboNumpy:
    def __init__(self, Q):
        if not self._is_valid_Q(Q):
            raise ValueError("Matrix Q must be square and symmetric.")
        self.Q = Q

    def size(self):
        """
        Returns the size of the QUBO matrix,
        which is the dimension of the square matrix.
        """
        return self.Q.shape[0]

    @staticmethod
    def _is_valid_Q(Q):
        """Check if the matrix Q is a valid QUBO matrix."""
        return (
            isinstance(Q, np.ndarray)
            and Q.ndim == 2
            and Q.shape[0] == Q.shape[1]
            and np.allclose(Q, Q.T)
        )

    def objective_value(self, x):
        """
        Calculate the objective value for a given binary vector x.
        """
        if len(x) != self.var_num:
            raise ValueError(
                "Binary vector x must have the same length as the dimensions of Q."
            )
        return x @ self.Q @ x

    @property
    def var_num(self):
        """Returns the number of variables in the QUBO matrix."""
        return self.Q.shape[0]
