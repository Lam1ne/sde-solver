"""
Run all examples and show convergence test
"""

import matplotlib.pyplot as plt
import numpy as np

from sde_solver.euler_maruyama import euler_maruyama


def example_convergence():
    """
    Convergence test - checking if this actually works lol

    Should get O(sqrt(dt)) strong convergence
    """
    print("\n" + "=" * 60)
    print("Convergence Test")
    print("=" * 60)

    X0 = 1.0
    mu = 0.5
    sigma = 0.3
    T = 1.0

    def drift(X, t):
        return mu * X

    def diffusion(X, t):
        return sigma * X

    # exact solution at T
    np.random.seed(123)
    dW_exact = np.random.randn()
    X_exact = X0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * dW_exact)

    N_values = [10, 50, 100, 500, 1000, 5000]
    errors = []

    print("\nTesting different step sizes:")
    for N in N_values:
        np.random.seed(123)
        t, X = euler_maruyama(X0, drift, diffusion, T, N, M=1)
        error = abs(X[0, -1] - X_exact)
        errors.append(error)
        print(f"N = {N:5d} : Error = {error:.6f}")

    # convergence plot
    plt.figure(figsize=(10, 6))
    plt.loglog(N_values, errors, "bo-", linewidth=2, markersize=8, label="Actual error")
    # theoretical convergence rate
    plt.loglog(
        N_values,
        [1 / np.sqrt(N) for N in N_values],
        "r--",
        linewidth=2,
        label="O(1/√N) theoretical",
    )
    plt.xlabel("Number of steps N")
    plt.ylabel("Absolute error")
    plt.title("Convergence of Euler-Maruyama")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    try:
        plt.savefig("euler_maruyama_convergence.png", dpi=150)
        print("\nPlot saved: euler_maruyama_convergence.png")
    except Exception as e:
        print("(couldn't save convergence plot)", e)
    plt.show()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SDE SOLVER - EULER-MARUYAMA SCHEME")
    print("Numerical Solution of SDEs")
    print("=" * 60)

    # Import and run examples
    from examples.black_scholes import run_black_scholes_simulation
    from examples.interest_rates import run_interest_rate_model

    run_black_scholes_simulation()
    run_interest_rate_model()
    example_convergence()

    print("\n" + "=" * 60)
    print("Done! Check the plots")
    print("=" * 60 + "\n")
