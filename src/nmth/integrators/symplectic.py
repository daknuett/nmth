"""
LeapFrog based symplectic integrators.
"""

class LeapFrog:
    """
    https://en.wikipedia.org/wiki/Leapfrog_integration
    """
    def __init__(self):
        self.coefficients = [
            (0, 1/2)
            , (1, 1/2)
        ]
    def __call__(self, X, P, dt, force, force_args):
        if not isinstance(X, list):
            for ci, di in self.coefficients:
                X = X + ci * dt * P
                P = di * dt * force(X, *force_args) + P
        else:
            for ci, di in self.coefficients:
                X = [Xj + ci * dt * Pj for Xj, Pj in zip(X,P)]
                P = [di * dt * Fj + Pj for Fj, Pj in zip(force(X, *force_args), P)]
        return X, P

class OM4(LeapFrog):
    def __init__(self):
        self.omega0 = - 2**(1/3) / (2 - 2**(1/3))
        self.omega1 = 1 / (2 - 2**(1/3))
        self.coefficients = [
            (self.omega1 / 2, self.omega1)
            , ((self.omega0 + self.omega1) / 2, self.omega0)
            , ((self.omega0 + self.omega1) / 2, self.omega1)
            , (self.omega1 / 2, 0)
        ]
class Yoshida8(LeapFrog):
    """
    Solution A of 8th order integrator from [1].
    
    1. Yoshida, H. Construction of higher order symplectic integrators. Physics Letters A 150, 262–268 (1990).

    """
    def __init__(self):
        self.ws = [
            0
            , -0.161582374150097E1
            , -0.244699182370524E1
            , -0.716989419708120E-2
            , 0.244002732616735E1
            , 0.157739928123617E0
            , 0.182020630970714E1
            , 0.104242620869991E1
        ]
        self.ws[0] = 1 - 2*sum(self.ws)
        
        d = list(reversed(self.ws[1:])) + self.ws + [0] # FIXME: I don't get the paper here. I will just set d_{16} to 0 and see what happens
        sumterms = [(wip1 + wi) / 2 for wip1, wi in zip(self.ws[1:], self.ws)] + [self.ws[-1] / 2]
        c = list(reversed(sumterms)) + sumterms
        self.coefficients = [(ci, di) for ci, di in zip(c, d)]
