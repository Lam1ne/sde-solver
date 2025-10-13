"""
Milstein scheme for SDEs - higher order convergence than Euler-Maruyama

TODO: implement this for better accuracy
The Milstein scheme adds a correction term that involves the derivative of the diffusion coefficient
Should get O(dt) strong convergence instead of O(sqrt(dt))
"""

import numpy as np


def milstein(X0, a, b, b_prime, T, N, M=1):
    """
    Milstein scheme - not implemented yet
    
    b_prime should be the derivative of b with respect to X
    """
