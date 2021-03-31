"""
Small dynamic programming library.
Import this in your modules/notebooks.
Authors: Alessandro Tenaglia, Roberto Masocco, Giacomo Solfizi
Date: March 31, 2021
"""

# Import necessary modules.
import numpy as np

def policy_eval(P, R, gamma, pi, v_pi, tol=1.0e-6):
    """Iteratively estimates the value of a given policy, starting from v."""
    # Build transition probability and reward matrices for the given policy.
    v = np.copy(v_pi)
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

def policy_improv(P, R, gamma, v):
    """Returns an improved policy from a given value."""
    # Initialize new policy vector.
    S = P.shape[0]
    A = R.shape[1]
    pi = np.zeros(S)
    # Policy improvement loop.
    q = np.zeros(A)
    for s in range(S):
        q *= 0.0
        for a in range(A):
            p_trans = P[s, :, a]
            q[a] = R[s, a] + gamma*np.dot(p_trans, v)
        pi[s] = np.argmax(q)
    return pi

def policy_iteration(P, R, gamma, v_init, pi_init, tol=1.0e-6):
    """Merges policy evaluation and improvement."""
    v = np.copy(v_init)
    prev_pi = np.copy(pi_init)
    # Policy iteration loops.
    while True:
        v = policy_eval(P, R, gamma, prev_pi, v, tol)
        pi = policy_improv(P, R, gamma, v)
        if np.linalg.norm(pi - prev_pi, ord=np.inf) == 0.0:
            break
        else:
            prev_pi = pi
    return pi, v

def value_iteration(P, R, gamma, v_init, tol=1.0e-6):
    """Returns the optimal policy and its value, from a given value."""
    # Initialize policy and value vectors.
    S = P.shape[0]
    A = R.shape[1]
    v_pi = np.zeros(S)
    q_pi = np.zeros(A)
    pi = np.zeros(S)
    prev_v = np.copy(v_init)
    # Value iteration loop.
    while True:
        for s in range(S):
            q_pi *= 0.0
            for a in range(A):
                p_trans = P[s, :, a]
                q_pi[a] = R[s, a] + gamma*np.dot(p_trans, prev_v)
            pi[s] = np.argmax(q_pi)
            v_pi[s] = q_pi[pi[s]]
        if np.linalg.norm(v_pi - prev_v) < tol:
            break
        else:
            prev_v = np.copy(v_pi)
    return pi, v_pi
