"""
Small dynamic programming library.
Import this in your modules/notebooks.
Authors: Alessandro Tenaglia, Roberto Masocco, Giacomo Solfizi
Date: March 31, 2021
"""

# Import necessary modules.
import numpy as np

def policy_eval(P, R, gamma, pi, v, tol=1.0e-6):
    """Iteratively estimates the value of a given policy, starting from v."""
    # Build transition probability and reward matrices for the given policy.
    S = P.shape[0]
    P_pi = np.zeros((S, S))
    R_pi = np.zeros(S)
    for s in range(S):
        P_pi[s, :] = np.copy(P[s, :, pi[s]])
        R_pi[s] = np.copy(R[s, pi[s]])
    # Policy evaluation loop.
    while True:
        prev_v = np.copy(v)
        v = R_pi + gamma*np.dot(P_pi, v)
        if np.linalg.norm(v - prev_v) < tol:
            break
    return v


