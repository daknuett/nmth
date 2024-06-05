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

def CG(A, b, x0, maxiter=1000, eps=1e-4):
    """
    Conjugate Gradient solver for symmetric matrices.

    Authors:
    - Daniel Knüttel 2024
    """
    if hasattr(A, "__call__"):
        apply_A = lambda x: A(x)
    else:
        apply_A = lambda x: A @ x
    rk = b - apply_A(x0)

    xk = x0
    pk = rk
    if np.linalg.norm(rk) < eps:
        return xk, {"converged": True, "k": 0}
        
    for k in range(maxiter):
        alpha = np.sum(np.conj(rk) * rk) / np.sum(np.conj(pk) * apply_A(pk))
        xkp1 = xk + alpha * pk
        rkp1 = rk - alpha * apply_A(pk)
        
        if np.linalg.norm(rk) < eps:
            return xk, {"converged": True, "k": k}
            
        beta = np.sum(np.conj(rkp1) * rkp1) / np.sum(np.conj(rk) * rk)
        pk = rkp1 + beta*pk
        rk = rkp1
        xk = xkp1
    return xk, {"converged": False, "k": k}

def GMRES(A, b, x0, maxiter=1000, eps=1e-4
              , innerproduct=None
              , prec=None):
    """
    GMRES solver using numpy backend.
    
    innerproduct is a function (vec,vec)->scalar which is a product.
    prec is a function vec->vec.

    Literature:
    - https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
    - https://www-users.cse.umn.edu/~saad/Calais/PREC.pdf

    Authors:
    - Daniel Knüttel 2024
    """
    if hasattr(A, "__call__"):
        apply_A = lambda x: A(x)
    else:
        apply_A = lambda x: A @ x

    if innerproduct is None:
        innerproduct = np.dot
    
    rk = b - apply_A(x0)

    rk_norm = np.linalg.norm(rk)
    if rk_norm <= eps:
        return x0, {"converged": True, "k": 0}

    vk = rk / rk_norm

    v = [None, vk]
    
    cs = np.zeros(maxiter + 2)
    sn = np.zeros(maxiter + 2)
    gamma = np.zeros(maxiter + 2)
    gamma[1] = rk_norm
    H = [None]
    
    converged = False
    for k in range(1, maxiter + 1):
        if prec is not None:
            z = prec(v[k])
        else:
            z = v[k]
        qk = apply_A(z)
        
        Hk = np.zeros(k + 2)
        for i in range(1, k + 1):
            Hk[i] = innerproduct(v[i], qk)
        for i in range(1, k + 1):
            qk -= Hk[i] * v[i]
            
        Hk[k+1] = np.linalg.norm(qk)

        for i in range(1, k):
            tmp = cs[i+1] * Hk[i] + sn[i+1] * Hk[i+1]
            Hk[i+1] = -sn[i+1] * Hk[i] + cs[i+1] * Hk[i+1]
            Hk[i] = tmp
            

        beta = np.sqrt(Hk[k]**2 + Hk[k + 1]**2)
        
        sn[k+1] = Hk[k+1] / beta
        cs[k+1] = Hk[k] / beta
        Hk[k] = beta
        
        
        gamma[k+1] = -sn[k+1] * gamma[k]
        gamma[k] = cs[k+1] * gamma[k]
        
        v.append(qk / Hk[k+1])
        H.append(Hk)
        if abs(gamma[k+1]) <= eps:
            converged = True
            break

    y = np.zeros(k+1)
    for i in reversed(range(1, k + 1)):
        overlap = 0
        for j in range(i+1, k+1):
            overlap += H[j][i] * y[j]
        y[i] = (gamma[i] - overlap) / H[i][i]
    if prec is None:
        x = x0 + sum(yi * vi for yi, vi in zip(y[1:], v[1:]))
    else:
        x = x0 + sum(yi * prec(vi) for yi, vi in zip(y[1:], v[1:]))
    return x, {"converged": converged, "k": k}



try:
    import tensorflow as tf

    def GMRES_tf(A, b, x0, maxiter=1000, eps=1e-4
                  , innerproduct=None
                  , prec=None):
        """
        GMRES solver using tensorflow backend.
        
        innerproduct is a function (vec,vec)->scalar which is a product.
        prec is a function vec->vec.

        Literature:
        - https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
        - https://www-users.cse.umn.edu/~saad/Calais/PREC.pdf

        Authors:
        - Daniel Knüttel 2024
        """
        if hasattr(A, "__call__"):
            apply_A = lambda x: A(x)
        else:
            apply_A = lambda x: A @ x

        if innerproduct is None:
            innerproduct = lambda x, y: tf.tensordot(tf.math.conj(x), y, len(x.shape)).numpy()

        norm = lambda x: tf.math.real(tf.norm(x, ord=2))
        
        rk = b - apply_A(x0)

        rk_norm = norm(rk)
        if rk_norm <= eps:
            return x0, {"converged": True, "k": 0}

        vk = rk / tf.cast(rk_norm, rk.dtype)

        v = [None, vk]
        
        cs = np.zeros(maxiter + 2)
        sn = np.zeros(maxiter + 2)
        gamma = np.zeros(maxiter + 2)
        gamma[1] = rk_norm
        H = [None]
        
        converged = False
        for k in range(1, maxiter + 1):
            if prec is not None:
                z = prec(v[k])
            else:
                z = v[k]
            qk = apply_A(z)
            
            Hk = np.zeros(k + 2)
            for i in range(1, k + 1):
                Hk[i] = innerproduct(v[i], qk)
            for i in range(1, k + 1):
                qk -= Hk[i] * v[i]
                
            Hk[k+1] = norm(qk)

            for i in range(1, k):
                tmp = cs[i+1] * Hk[i] + sn[i+1] * Hk[i+1]
                Hk[i+1] = -sn[i+1] * Hk[i] + cs[i+1] * Hk[i+1]
                Hk[i] = tmp
                

            beta = np.sqrt(Hk[k]**2 + Hk[k + 1]**2)
            
            sn[k+1] = Hk[k+1] / beta
            cs[k+1] = Hk[k] / beta
            Hk[k] = beta
            
            
            gamma[k+1] = -sn[k+1] * gamma[k]
            gamma[k] = cs[k+1] * gamma[k]
            
            v.append(qk / Hk[k+1])
            H.append(Hk)
            if abs(gamma[k+1]) <= eps:
                converged = True
                break

        y = np.zeros(k+1)
        for i in reversed(range(1, k + 1)):
            overlap = 0
            for j in range(i+1, k+1):
                overlap += H[j][i] * y[j]
            y[i] = (gamma[i] - overlap) / H[i][i]
        if prec is None:
            x = x0 + sum(yi * vi for yi, vi in zip(y[1:], v[1:]))
        else:
            x = x0 + sum(yi * prec(vi) for yi, vi in zip(y[1:], v[1:]))
        return x, {"converged": converged, "k": k}
except:
    pass
