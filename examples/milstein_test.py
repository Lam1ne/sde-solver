"""
Comparison between Euler-Maruyama and Milstein schemes
"""

import sys
import os
# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from sde_solver.euler_maruyama import euler_maruyama
from sde_solver.milstein import milstein

def run_milstein_comparison():
    """
    Compare Euler and Milstein convergence on GBM
    """
    print("\n" + "=" * 60)
    print("Milstein vs Euler Comparison")
    print("=" * 60)
    
    # GBM Parameters
    X0 = 1.0
    mu = 0.5
    sigma = 0.3
    T = 1.0
    
    def drift(X, t):
        return mu * X
    
    def diffusion(X, t):
        return sigma * X
        
    def diffusion_prime(X, t):
        return sigma * np.ones_like(X) # derivative of sigma*X is sigma
    
    # Exact solution for reference
    # We need to be careful with random seeds to compare properly
    # But for strong convergence we usually compare against a very fine path
    # Here we'll just check error against exact solution at T for a single path (weak-ish check but okay for demo)
    
    N_values = [10, 50, 100, 500, 1000]
    euler_errors = []
    milstein_errors = []
    
    print("Testing convergence...")
    
    for N in N_values:
        np.random.seed(42) # Same seed for all N to keep it somewhat consistent
        
        # Exact
        dW_total = np.random.randn() * np.sqrt(T) # This is wrong for pathwise comparison
        # For proper strong convergence we need to simulate the Brownian path
        # But let's just do a simple check:
        # We'll use the same Brownian increments for both schemes
        
        dt = T/N
        np.random.seed(42)
        dW = np.sqrt(dt) * np.random.randn(1, N)
        W_T = np.sum(dW)
        
        X_exact = X0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * W_T)
        
        # Euler
        np.random.seed(42) # Reset seed to generate same dW inside function
        _, X_euler = euler_maruyama(X0, drift, diffusion, T, N, M=1)
        euler_errors.append(abs(X_euler[0, -1] - X_exact))
        
        # Milstein
        np.random.seed(42) # Reset seed
        _, X_milstein = milstein(X0, drift, diffusion, diffusion_prime, T, N, M=1)
        milstein_errors.append(abs(X_milstein[0, -1] - X_exact))
        
        print(f"N={N:4d} | Euler Err: {euler_errors[-1]:.6f} | Milstein Err: {milstein_errors[-1]:.6f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.loglog(N_values, euler_errors, 'b-o', label='Euler-Maruyama (Order 0.5)')
    plt.loglog(N_values, milstein_errors, 'r-s', label='Milstein (Order 1.0)')
    
    # Reference lines
    plt.loglog(N_values, [1/np.sqrt(N) for N in N_values], 'b--', alpha=0.3, label='Slope 0.5')
    plt.loglog(N_values, [1/N for N in N_values], 'r--', alpha=0.3, label='Slope 1.0')
    
    plt.xlabel('Steps N')
    plt.ylabel('Error at T')
    plt.title('Convergence Comparison: Euler vs Milstein')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    
    try:
        plt.savefig('milstein_convergence.png', dpi=150)
        print("\nPlot saved: milstein_convergence.png")
    except:
        print("Could not save plot")
    plt.show()

if __name__ == "__main__":
    run_milstein_comparison()
