# SDE Solver

My implementation of numerical methods for solving Stochastic Differential Equations (SDEs).

Currently implements the **Euler-Maruyama scheme** with plans to add Milstein method.

## Description

The Euler-Maruyama scheme is a numerical method for solving SDEs of the form:

```math
dX(t) = a(X(t), t) dt + b(X(t), t) dW(t)
```

where:
- **a(X, t)** is the drift term
- **b(X, t)** is the diffusion term
- **W(t)** is a standard Brownian motion (Wiener process)

It's basically the stochastic version of Euler's method

## The Algorithm

The Euler-Maruyama discretization:

```math
X_{n+1} = X_n + a(X_n, t_n) Δt + b(X_n, t_n) ΔW_n
```

where Δt is the time step and ΔW_n ~ N(0, Δt) are Brownian increments

## Project Structure

```
sde-solver/
├── src/sde-solver
│   ├── euler_maruyama.py    # Main solver implementation
│   └── milstein.py          # Milstein scheme (TODO)
├── examples/
│   ├── black_scholes.py     # Stock price modeling
│   └── interest_rates.py    # Interest rate modeling
├── tests/
│   └── test_euler_maruyama.py
├── run_all.py               # Run all examples
└── pyproject.toml
```

## Installation

In order to run examples and tests, simply use [uv](https://docs.astral.sh/uv/).

It can be installed using:
```bash
pipx install uv
```

To install the `sde-solver` package, use:
```bash
pip install .
```


## Usage

### Quick Start

Run all examples at once:
```bash
uv run run_all.py
```

### Individual Examples

Black-Scholes stock price model:
```bash
uv run examples/black_scholes.py
```

Interest rate model:
```bash
uv run examples/interest_rates.py
```

### Using in Your Code

```python
from sde_solver.euler_maruyama import euler_maruyama

# Define drift and diffusion
def drift(X, t):
    return mu * X

def diffusion(X, t):
    return sigma * X

# Solve the SDE
t, X = euler_maruyama(X0=1.0, a=drift, b=diffusion, T=1.0, N=1000, M=5)
```

### Run Tests

```bash
uv run pytest
```

## Examples Included

### 1. Black-Scholes Model (`examples/black_scholes.py`)
Geometric Brownian Motion for stock prices:
```
dS(t) = μ S(t) dt + σ S(t) dW(t)
```

### 2. Interest Rate Model (`examples/interest_rates.py`)
Ornstein-Uhlenbeck mean-reverting process:
```
dX(t) = θ(μ - X(t)) dt + σ dW(t)
```

### 3. Convergence Analysis (`run_all.py`)
Testing convergence rate O(√Δt)

## Results

The program makes three plots:
- geometric_brownian_motion.png
- ornstein_uhlenbeck_process.png
- euler_maruyama_convergence.png

## Theory

### Convergence
Euler-Maruyama has:
- **Strong convergence**: order 0.5
- **Weak convergence**: order 1.0

Tested in example 3

### Applications
- Finance (option pricing)
- Physics (Brownian motion)
- Biology (population models)

## Parameters

```python
euler_maruyama(X0, a, b, T, N, M=1)
```

- X0: initial condition
- a: drift function a(X, t)
- b: diffusion function b(X, t)
- T: final time
- N: number of time steps
- M: number of paths (default 1)

## References

- Kloeden & Platen - "Numerical Solution of Stochastic Differential Equations with Jumps in Finance"
- Wikipedia article on the Euler-Maruyama method





