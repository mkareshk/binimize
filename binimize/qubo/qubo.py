class QUBO:
    def __new__(cls, Q, backend="numpy", device="cpu"):
        if backend == "torch":
            from .torch import QuboTorch

            return QuboTorch(Q, device=device)

        elif backend == "numpy":
            from .numpy import QuboNumpy

            return QuboNumpy(Q)

        else:
            raise ValueError("Unsupported backend. Choose 'numpy' or 'torch'.")
