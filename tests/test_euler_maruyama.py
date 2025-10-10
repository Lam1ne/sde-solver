"""
Basic tests for the Euler-Maruyama solver

TODO: add proper unit tests with pytest later
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from euler_maruyama import euler_maruyama


def test_basic_simulation():
    """Test that basic simulation runs without errors"""
    
    def drift(X, t):
        return 0.1 * X
    
    def diffusion(X, t):
        return 0.2 * X
    
    t, X = euler_maruyama(X0=1.0, a=drift, b=diffusion, T=1.0, N=100, M=3)
    
    assert t.shape == (101,), "Time array has wrong shape"
    assert X.shape == (3, 101), "Solution array has wrong shape"
    assert X[0, 0] == 1.0, "Initial condition not satisfied"
    
    print("✓ Basic simulation test passed")


def test_deterministic_case():
    """Test with zero diffusion (should be deterministic ODE)"""
    
    def drift(X, t):
        return X  # dX/dt = X, solution is X(t) = X0 * exp(t)
    
    def diffusion(X, t):
        return 0.0 * X  # no randomness
    
    np.random.seed(42)
    t, X = euler_maruyama(X0=1.0, a=drift, b=diffusion, T=1.0, N=1000, M=1)
    
    # Check against exact solution
    X_exact = np.exp(t)
    error = np.abs(X[0, -1] - X_exact[-1])
    
    assert error < 0.01, f"Error too large in deterministic case: {error}"
    
    print("✓ Deterministic case test passed")


if __name__ == "__main__":
    print("Running tests...")
    test_basic_simulation()
    test_deterministic_case()
    print("\nAll tests passed!")
