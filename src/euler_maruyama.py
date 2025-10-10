"""
Euler-Maruyama Scheme Implementation for Stochastic Differential Equations (SDEs)

This code solves SDEs of the form:
    dX(t) = a(X(t), t) dt + b(X(t), t) dW(t)
    
where:
    - a(X, t) is the drift term
    - b(X, t) is the diffusion term  
    - W(t) is a Brownian motion (Wiener process)
"""

import numpy as np

# TODO: maybe add Milstein scheme later for better convergence?
DEBUG = False  # quick flag for printing stuff while debugging


def euler_maruyama(X0, a, b, T, N, M=1):
    """
    Euler-Maruyama scheme to solve an SDE numerically
    
    This is the main function I implemented for my project.
    It's basically the stochastic version of Euler's method.
    
    Parameters:
    X0 - initial condition
    a - drift function a(X, t)
    b - diffusion function b(X, t)  
    T - final time
    N - number of time steps
    M - number of paths (default 1)
    
    Returns: t, X where X has shape (M, N+1)
    """
    dt = T / N  
    t = np.linspace(0, T, N + 1)
    
    # Initialize paths
    X = np.zeros((M, N + 1))
    X[:, 0] = X0
    
    # Generate Brownian increments - this is the random part!
    dW = np.sqrt(dt) * np.random.randn(M, N)
    
    # Main loop
    for i in range(N):
        X[:, i + 1] = X[:, i] + a(X[:, i], t[i]) * dt + b(X[:, i], t[i]) * dW[:, i]
        # if DEBUG and (i % 200 == 0 or i == N-1):
        #     print('step', i, 'mean(X)=', float(X[:, i].mean()))
    
    return t, X
