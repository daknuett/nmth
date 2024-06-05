#
# Copyright(c) 2024 Daniel Knüttel
#

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
