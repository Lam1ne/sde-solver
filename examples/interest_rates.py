"""
Interest rate modeling using Ornstein-Uhlenbeck process

Mean-reverting model for interest rates:
dX(t) = θ(μ - X(t)) dt + σ dW(t)
"""

import matplotlib.pyplot as plt
import numpy as np

from sde_solver.euler_maruyama import euler_maruyama


def run_interest_rate_model():
    """
    Example: Ornstein-Uhlenbeck Process

    Mean-reverting process - models things that return to an average
    dX(t) = θ(μ - X(t)) dt + σ dW(t)

    θ = speed of mean reversion
    μ = long-term mean
    σ = volatility
    """
    print("=" * 60)
    print("Interest Rate Model: Ornstein-Uhlenbeck Process")
    print("=" * 60)

    # params
    X0 = 0.0
    theta = 1.0
    mu = 1.5
    sigma = 0.3
    T = 5.0
    N = 1000
    M = 10

    def drift(X, t):
        return theta * (mu - X)

    def diffusion(X, t):
        return sigma * np.ones_like(X)

    t, X = euler_maruyama(X0, drift, diffusion, T, N, M)

    # plots
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for i in range(M):
        plt.plot(t, X[i, :], alpha=0.6)
    plt.axhline(y=mu, color="r", linestyle="--", linewidth=2, label=f"Mean μ={mu}")
    plt.xlabel("Time")
    plt.ylabel("X(t)")
    plt.title("Ornstein-Uhlenbeck Process\n" + f"X0={X0}, θ={theta}, μ={mu}, σ={sigma}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(X[:, -1], bins=30, density=True, alpha=0.7, edgecolor="black")
    plt.xlabel("X(T)")
    plt.ylabel("Density")
    plt.title(f"Distribution at t={T}")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    try:
        plt.savefig("ornstein_uhlenbeck_process.png", dpi=150)
        print("\nPlot saved: ornstein_uhlenbeck_process.png")
    except Exception as e:
        print("(couldn't save OU plot)", e)
    plt.show()

    return t, X


if __name__ == "__main__":
    run_interest_rate_model()
