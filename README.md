# binimize: A Python Library for QUBO Problems

binimize is a Python library designed to solve Quadratic Unconstrained Binary Optimization (QUBO) problems efficiently using parallel computation on both CPU and GPU. It provides a flexible approach to handle these problems using either NumPy or PyTorch, making it suitable for a variety of computational environments.

## Badges

![Build Status](https://github.com/yourusername/binimize/actions/workflows/python-package.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)
![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)

## Installation

To install binimize, you need to have Python 3.10 or higher. It's recommended to use a virtual environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/binimize.git
cd binimize

# Install using pip
pip install .
```

## Dependencies

- NumPy
- Pandas
- PyTorch
- Joblib

These dependencies will be automatically installed when you install the library.

## Usage

### Creating a QUBO Instance

You can create a QUBO problem instance using either NumPy or PyTorch backends.

```python
import numpy as np
from binimize import QUBO

# Define the Q matrix using NumPy
Q_np = np.array([[1, 2], [2, 1]])

# Create a QUBO instance with NumPy backend
qubo_numpy = QUBO(Q_np, backend="numpy")
```

Or, for a PyTorch backend:

```python
import torch
from binimize import QUBO

# Define the Q matrix using PyTorch
Q_torch = torch.tensor([[1, 2], [2, 1]], dtype=torch.float32)

# Create a QUBO instance with PyTorch backend, optionally using GPU
qubo_torch = QUBO(Q_torch, backend="torch", device="cuda")
```

### Solving the QUBO Problem

To solve the QUBO problem and find the objective values for all possible combinations of binary variables:

```python
from binimize import ExactSolver

# Initialize the solver, specify the number of jobs for parallel computation
solver = ExactSolver(n_jobs=4)

# Solve the QUBO problem
results_df = solver.solve(qubo_numpy)  # or qubo_torch
print(results_df)
```

## Output

The output will be a Pandas DataFrame with each row representing a binary combination and its corresponding QUBO value:

```
   x_1  x_2  value
0    0    0      0
1    0    1      2
2    1    0      1
3    1    1      6
```

## Testing

To run tests:

```bash
pytest tests/
```

Ensure that you have pytest installed and that your tests are located in the `tests/` directory.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests with any enhancements. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.
