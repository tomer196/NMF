# General Proximal Method Solver

A modular framework for solving optimization problems using proximal methods, with examples for LASSO regression (ISTA/FISTA) and Non-negative Matrix Factorization (NMF) with regularization.

## Overview

This framework provides a flexible and extensible solver for optimization problems of the form:

```
minimize f(x) + g(x)
```

where:
- `f(x)` is a smooth function with computable gradient
- `g(x)` is a possibly non-smooth function with a computable proximal operator

## Features

- **Modular Design**: Easily plug in different objectives and proximal operators
- **Multiple Solvers**:
  - Standard Proximal Gradient (ISTA)
  - Accelerated Proximal Gradient (FISTA)
  - Alternating Proximal Gradient (for multi-variable problems)
- **Example Implementations**:
  - LASSO regression with L1 regularization
  - Non-negative Matrix Factorization with regularization

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Example 1: LASSO Regression with ISTA

Solves: `minimize (1/2)||Ax - b||^2 + λ||x||_1`

```bash
python lasso_ista_example.py
```

This example:
- Generates synthetic sparse data
- Compares ISTA and FISTA algorithms
- Visualizes convergence and recovered coefficients

### Example 2: NMF with Regularization

Solves: `minimize (1/2)||A - WH||_F^2 + λ_W||W||_1 + λ_H||H||_1` subject to `W ≥ 0, H ≥ 0`

```bash
python nmf_proximal_example.py
```

This example:
- Generates synthetic non-negative matrix data
- Uses alternating proximal gradient method
- Visualizes matrix reconstruction and convergence

## Framework Structure

### Core Components

#### 1. `Objective` (Abstract Base Class)
```python
class Objective(ABC):
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the objective function"""
        pass
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient"""
        pass
```

#### 2. `ProximalOperator` (Abstract Base Class)
```python
class ProximalOperator(ABC):
    def apply(self, x: np.ndarray, step_size: float) -> np.ndarray:
        """Apply the proximal operator"""
        pass
```

#### 3. Solvers
- `ProximalGradientSolver`: Standard ISTA
- `AcceleratedProximalGradientSolver`: FISTA with Nesterov acceleration
- `AlternatingProximalSolver`: For problems with multiple variables

## Creating Custom Problems

### Step 1: Define Your Objective

```python
from proximal_solver import Objective

class MyObjective(Objective):
    def __init__(self, ...):
        # Initialize parameters
        pass
    
    def evaluate(self, x):
        # Compute f(x)
        return objective_value
    
    def gradient(self, x):
        # Compute ∇f(x)
        return gradient
```

### Step 2: Define Your Proximal Operator

```python
from proximal_solver import ProximalOperator

class MyProxOperator(ProximalOperator):
    def __init__(self, lambda_reg):
        self.lambda_reg = lambda_reg
    
    def apply(self, x, step_size):
        # Compute prox_{step_size*g}(x)
        return proximal_result
```

### Step 3: Solve Your Problem

```python
from proximal_solver import ProximalGradientSolver

# Create instances
objective = MyObjective(...)
prox_op = MyProxOperator(lambda_reg=0.1)

# Initialize solver
solver = ProximalGradientSolver(
    objective=objective,
    prox_operator=prox_op,
    step_size=0.01,
    max_iter=1000,
    tol=1e-6,
    verbose=True
)

# Solve
x_init = np.zeros(n)
x_optimal = solver.solve(x_init)
```

## Common Proximal Operators

### L1 Norm (Soft Thresholding)
`prox_{λ||·||_1}(x) = sign(x) ⊙ max(|x| - λ, 0)`

### Non-negative Constraint
`prox_{I_+(·)}(x) = max(x, 0)`

### L2 Norm (Shrinkage)
`prox_{λ||·||_2}(x) = x · max(1 - λ/||x||_2, 0)`

### Combined L1 + Non-negativity
`prox(x) = max(sign(x) ⊙ max(|x| - λ, 0), 0)`

## Algorithm Details

### ISTA (Iterative Shrinkage-Thresholding Algorithm)
```
x^(k+1) = prox_{step_size·g}(x^k - step_size·∇f(x^k))
```

### FISTA (Fast ISTA)
Uses Nesterov momentum for accelerated convergence:
```
y^k = x^k + ((t^k - 1)/t^(k+1)) · (x^k - x^(k-1))
x^(k+1) = prox_{step_size·g}(y^k - step_size·∇f(y^k))
```

### Alternating Minimization
For problems with multiple variables (W, H, ...):
```
W^(k+1) = argmin_W f(W, H^k)
H^(k+1) = argmin_H f(W^(k+1), H)
```

## Performance Tips

1. **Step Size Selection**: Use `step_size = 1/L` where L is the Lipschitz constant of ∇f
2. **Initialization**: Good initialization can significantly speed up convergence
3. **Use FISTA**: When applicable, FISTA typically converges faster than ISTA
4. **Precompute**: Cache expensive matrix products (like A^T·A)

## Mathematical Background

### Proximal Operator
The proximal operator of a function g is defined as:
```
prox_g(v) = argmin_x { (1/2)||x - v||^2 + g(x) }
```

It generalizes the projection operator and is the key computational step in proximal methods.

### Convergence
- ISTA: O(1/k) convergence rate
- FISTA: O(1/k²) convergence rate (optimal for first-order methods)

## References

1. Beck, A., & Teboulle, M. (2009). "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems." SIAM Journal on Imaging Sciences.

2. Parikh, N., & Boyd, S. (2014). "Proximal Algorithms." Foundations and Trends in Optimization.

3. Lee, D. D., & Seung, H. S. (1999). "Learning the parts of objects by non-negative matrix factorization." Nature.

## License

MIT License

## Author

Proximal Method Solver Framework
