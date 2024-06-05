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

"""
Integrators of the Runge-Kutta type
"""


class runge_kutta:
    """
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """
    def __init__(self, weights, nodes, rk_matrix):
        self.weights = weights
        self.nodes = nodes
        self.rk_matrix = rk_matrix

    def __call__(self, Y, t, dt, F, F_args):
        """
        integrates 

            d/dt y Y(t) = F(t, Y(t), *F_args)

        from t to t+dt
        """
        ki = []
        for i, ci in enumerate(self.nodes):
            ki.append(F(t + ci * dt
                        , Y + self.contract_rk_matrix(ki) * dt
                        , *F_args))

        result = Y + dt * sum(b * k for b,k in zip(self.weights, ki))
        return result

    def contract_rk_matrix(self, ki):
        j = len(ki)
        return sum(self.rk_matrix[j][i] * k for i, k in enumerate(ki))

class RK4(runge_kutta):
    """
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Examples
    """
    def __init__(self):
        self.weights = [1. / 6, 1. / 3, 1. / 3, 1. / 6]
        self.nodes = [0, 0.5, 0.5, 1]
        self.rk_matrix = [[0.0], [0.5, 0], [0, 0.5, 0], [0, 0, 1, 0]]


class SSPRK3(runge_kutta):
    """
    https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
    """
    def __init__(self):
        self.weights = [1/6, 1/6, 2/3]
        self.nodes = [0, 1, 1/2]
        self.rk_matrix = [[0.0], [1.0, 0.0], [0.25, 0.25, 0.0]]

class Euler1(runge_kutta):
    def __init__(self):
        self.weights = [1.0]
        self.nodes = [0.0]
        self.rk_matrix = [[0.0]]

