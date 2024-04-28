from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from binimize.qubo import QuboNumpy, QuboTorch


class ExactSolver:
    """
    Efficiently solves QUBO problems using parallel computation on CPU or GPU.
    """

    def __init__(self, n_jobs: int = 1) -> None:
        self.n_jobs = n_jobs

    def solve(self, qubo: Any) -> pd.DataFrame:
        """
        Solves the QUBO problem for all binary combinations and returns results in a DataFrame.
        """
        self.qubo = qubo
        num_vars = qubo.var_num
        combinations = list(product([0, 1], repeat=num_vars))

        if isinstance(qubo, QuboNumpy):
            results = self._solve_numpy(combinations)
        elif isinstance(qubo, QuboTorch):
            results = self._solve_torch(combinations)

        columns = [f"x_{i+1}" for i in range(num_vars)] + ["value"]
        return pd.DataFrame(results, columns=columns)

    def _solve_numpy(self, combinations):
        """
        Helper to solve QUBO problem using QuboNumpy with parallel computation.
        """
        return Parallel(n_jobs=self.n_jobs)(
            delayed(self._compute_value)(np.array(combo)) for combo in combinations
        )

    def _solve_torch(self, combinations):
        """
        Helper to solve QUBO problem using QuboTorch with parallel computation.
        """
        return Parallel(n_jobs=self.n_jobs)(
            delayed(self._compute_value)(torch.tensor(combo, device=self.qubo.device))
            for combo in combinations
        )

    def _compute_value(self, x):
        """
        Compute the QUBO objective value for a given binary vector x.
        """
        value = self.qubo.objective_value(x)
        return (
            list(x.cpu().numpy()) if isinstance(x, torch.Tensor) else list(x) + [value]
        )
