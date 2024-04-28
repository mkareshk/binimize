from typing import Literal

import numpy as np
from torch import Tensor


class QUBO:
    def __new__(
        cls,
        Q: Tensor | np.ndarray,
        backend: Literal["numpy", "torch"] = "numpy",
        device: Literal["cpu", "cuda"] = "cpu",
    ):
        if backend == "torch":
            from .torch import QuboTorch

            return QuboTorch(Q, device=device)

        elif backend == "numpy":
            from .numpy import QuboNumpy

            return QuboNumpy(Q)

        else:
            raise ValueError("Unsupported backend. Choose 'numpy' or 'torch'.")
