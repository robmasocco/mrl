"""
Small dynamic programming library.
Import this in your modules/notebooks.
Authors: Alessandro Tenaglia, Roberto Masocco, Giacomo Solfizi
Date: March 31, 2021
"""

# Import necessary modules.
import numpy as np

def policy_eval(P, R, gamma, pi, v_init, tol=1.0e-6):
    """Iteratively estimates the value of a given policy, starting from v."""
    # Number of states.
    S = P.shape[0]
    # Transition matrix.
    P_pi = np.zeros((S, S))
    # Expected rewards.
    R_pi = np.zeros(S)
    # Build transition probability and reward matrices for the given policy.
    for s in range(S):
        P_pi[s, :] = np.copy(P[s, :, pi[s]])
        R_pi[s] = np.copy(R[s, pi[s]])
    # Policy evaluation loop.
    v = np.copy(v_init)
    while True:
        prev_v = np.copy(v)
        v = R_pi + gamma*np.dot(P_pi, prev_v)
        if np.linalg.norm(v - prev_v) < tol:
            break
    return v

def policy_improv(P, R, gamma, v_pi):
    """Returns an improved policy from a given value."""
    # Number of states.
    S = P.shape[0]
    # Number of actions.
    A = R.shape[1]
    # Initialize new policy vector.
    pi = np.zeros(S, dtype=np.int64)
    # Policy improvement loop.
    q = np.zeros(A)
    for s in range(S):
        q.fill(0.0)
        for a in range(A):
            p_trans = P[s, :, a]
            q[a] = R[s, a] + gamma*np.dot(p_trans, v_pi)
        pi[s] = np.argmax(q)
    return pi

def policy_iteration(P, R, gamma, v_init, pi_init, tol=1.0e-6):
    """Merges policy evaluation and improvement."""
    # Initial estimate of the policy.
    prev_pi = np.copy(pi_init)
    # Initial estimate of the policy value.
    v = np.copy(v_init)
    # Policy iteration loops.
    while True:
        v = policy_eval(P, R, gamma, prev_pi, v, tol)
        pi = policy_improv(P, R, gamma, v)
        if np.linalg.norm(pi - prev_pi, ord=np.inf) == 0.0:
            break
        else:
            prev_pi = np.copy(pi)
    return pi, v

def value_iteration(P, R, gamma, v_init, tol=1.0e-6):
    """Returns the optimal policy and its value, from a given value."""
    # Number of states.
    S = P.shape[0]
    # Number of actions.
    A = R.shape[1]
    # Initialize policy vector.
    pi = np.zeros(S, dtype=np.int8)
    # Initialize policy value vectors.
    v = np.zeros(S)
    q = np.zeros(A)
    # Value iteration loop.
    prev_v = np.copy(v_init)
    while True:
        for s in range(S):
            q.fill(0.0)
            for a in range(A):
                p_trans = P[s, :, a]
                q[a] = R[s, a] + gamma*np.dot(p_trans, prev_v)
            pi[s] = np.argmax(q)
            v[s] = q[pi[s]]
        temp = v - prev_v
        print(np.linalg.norm(v - prev_v),  np.count_nonzero(temp != 0.0))
        if np.linalg.norm(v - prev_v) < tol:
            break
        else:
            prev_v = np.copy(v)
    return pi, v
