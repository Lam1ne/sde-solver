"""
Milstein scheme for SDEs - higher order convergence than Euler-Maruyama

The Milstein scheme adds a correction term that involves the derivative of the diffusion coefficient
Should get O(dt) strong convergence instead of O(sqrt(dt))
"""

import numpy as np


def milstein(X0, a, b, b_prime, T, N, M=1):
    """
    Milstein scheme implementation
    
    This one is better than Euler-Maruyama because it has strong order 1.0
    (Euler only has 0.5).
    
    We need the derivative of b(X) though!
    
    Parameters:
    b_prime - derivative of b with respect to X: db/dX
    """
    dt = T / N
    t = np.linspace(0, T, N + 1)
    
    # Initialize paths
    X = np.zeros((M, N + 1))
    X[:, 0] = X0
    
    # Generate Brownian increments
    dW = np.sqrt(dt) * np.random.randn(M, N)
    
    # Main loop
    for i in range(N):
        # Evaluate functions
        X_val = X[:, i]
        t_val = t[i]
        
        drift = a(X_val, t_val)
        diffusion = b(X_val, t_val)
        diffusion_prime = b_prime(X_val, t_val)
        
        # Milstein correction: 0.5 * b * b' * (dW^2 - dt)
        # This is the extra term compared to Euler!
        correction = 0.5 * diffusion * diffusion_prime * (dW[:, i]**2 - dt)
        
        X[:, i + 1] = X_val + drift * dt + diffusion * dW[:, i] + correction
        
        # Debugging: check if correction is too large?
        # if np.max(np.abs(correction)) > 1.0:
        #     print("Warning: large correction term at step", i)
    
    return t, X
