"""
Black-Scholes model example using Geometric Brownian Motion

Stock price modeling with the famous GBM equation:
dS(t) = μ S(t) dt + σ S(t) dW(t)
"""

import matplotlib.pyplot as plt
import numpy as np

from sde_solver.euler_maruyama import euler_maruyama


def run_black_scholes_simulation():
    """
    Example: Geometric Brownian Motion (Black-Scholes model)

    This is the famous model used in finance!
    dS(t) = μ S(t) dt + σ S(t) dW(t)

    where μ is the drift rate and σ is the volatility
    """
    print("=" * 60)
    print("Black-Scholes Model: Geometric Brownian Motion")
    print("=" * 60)

    # Parameters - using typical stock market values
    S0 = 100.0
    mu = 0.1  # 10% return
    sigma = 0.2  # 20% volatility
    T = 1.0
    N = 1000
    M = 5
    
    # TODO: maybe load parameters from a config file later?

    # drift and diffusion functions
    def drift(S, t):
        return mu * S

    def diffusion(S, t):
        return sigma * S

    # simulate
    t, S = euler_maruyama(S0, drift, diffusion, T, N, M)

    # Exact solution for comparison (from textbook p.342)
    np.random.seed(42)
    W = np.cumsum(np.random.randn(N) * np.sqrt(T / N))
    S_exact = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * np.concatenate(([0], W)))

    # plotting
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for i in range(M):
        plt.plot(t, S[i, :], alpha=0.7, label=f"Path {i + 1}")
    plt.xlabel("Time (years)")
    plt.ylabel("Price S(t)")
    plt.title("Geometric Brownian Motion\n" + f"S0={S0}, μ={mu}, σ={sigma}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(t, S[0, :], "b-", alpha=0.7, label="Euler-Maruyama")
    plt.plot(t, S_exact, "r--", alpha=0.7, label="Exact solution")
    plt.xlabel("Time (years)")
    plt.ylabel("Price S(t)")
    plt.title("Comparison with Exact Solution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    try:
        plt.savefig("geometric_brownian_motion.png", dpi=150)
        print("\nPlot saved: geometric_brownian_motion.png")
    except Exception as e:
        # sometimes saving fails on weird environments; not a big deal for class
        print("(couldn't save GBM plot)", e)
    plt.show()

    return t, S


if __name__ == "__main__":
    run_black_scholes_simulation()
