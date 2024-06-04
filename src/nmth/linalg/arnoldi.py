import numpy as np

"""
Arnoldi iteration.
"""

def arnoldi_k(A, qprev, Hprev, k):
    if hasattr(A, "__call__"):
        qk = A(qprev[k-1])
    else:
        qk = A @ qprev[k-1]

    for j in range(k):
        Hprev[j, k-1] = np.sum(qprev[j] * qk)
        qk = qk - Hprev[j, k-1] * qprev[j]
    Hprev[k, k-1] = np.linalg.norm(qk)
    qk /= Hprev[k, k-1]
    qprev[k] = qk
    return qprev, Hprev

def arnoldi(A, n, q0):
    if n < 1:
        raise ValueError("nothing to do")
    H = np.zeros((n+1, n))
    # This is Q^T because the shape of the
    # vectors is not known.
    q = np.zeros((n+1, *(q0.shape)))

    q[0] = q0

    for k in range(1, n+1):
        q, H = arnoldi_k(A, q, H, k)

    return q, H
