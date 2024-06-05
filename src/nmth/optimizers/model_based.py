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

def adam(model, Winit, vin, b
         , eps=1e-8, alpha=1e-3, beta1=0.9, beta2=0.999, maxiter=10_000):
    r"""

    ```model`` must have the method ``dcost``::
        
        model.dcost(W, vin, b)

    which computes

    .. math::

        \partial_W model(W, vin, b)

    @misc{kingma2017adam,
          title={Adam: A Method for Stochastic Optimization}, 
          author={Diederik P. Kingma and Jimmy Ba},
          year={2017},
          eprint={1412.6980},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }
    """

    m = [0 for w in Winit]
    v = [0 for w in Winit]
    W = [np.copy(w) for w in Winit]

    for t in range(1, maxiter + 1):
        g = model.dcost(W, vin, b)

        if(sum(np.sum(dw**2) for dw in g) < eps):
            return W, (True, t)

        m = [beta1 * mi + (1 - beta1) * gi for mi,gi in zip(m, g)]
        v = [beta2 * vi + (1 - beta2) * gi**2 for vi,gi in zip(v, g)]
        mhat = [mi / (1 - beta1**t) for mi in m]
        vhat = [vi / (1 - beta1**t) for vi in v]

        W = [w - alpha*mi / (np.sqrt(vi) + eps) for w, (mi,vi) in zip(W, zip(mhat,vhat))]

    return W, (False, t)
