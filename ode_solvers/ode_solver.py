import autograd.numpy as np
from autograd import grad, jacobian
from scipy import linalg
from typing import Callable
import numpy.typing as npt

from ode_models import *


class ODESolver:
    """ODESolver superclass

        ODE:
        u' = f(u, t)
        u(t_0) = U_t0
    """

    def __init__(self, ode_problem: ODEModel, tolerance: float):
        self.f = ode_problem.f
        self.y0 = ode_problem.y0.astype(float)  # initial condition

        self.u = None  # solution
        self.i = None  # current number of step iteration

        self.h = (ode_problem.T - ode_problem.t0) / (ode_problem.number_of_points_to_discretization + 1)
        self.t = np.linspace(ode_problem.t0, ode_problem.T, ode_problem.number_of_points_to_discretization + 2)  # array of time points corresponding to solution

        self.tol = tolerance
        self.m = len(self.y0)
        self.s = len(self.b)
        
    def step(self):
        ti, yi = self.t[0], self.y0  # initial condition
        current_time_point = ti
        yield ti , np.array(yi)  # first point (begging point)
        for ti in self.t[1:]:
            yi += self.h * self.phi(current_time_point, yi)
            current_time_point = ti
            yield ti , np.array(yi)
        
    def solve(self):
        return np.array(list(self.step()))

    def phi(self, current_time_step, y0):
        """Advance solution one time step."""
        raise NotImplementedError
    
    def solve_(self):
        n = self.t.size
        n_of_eqns = self.y0.size
        self.u = np.zeros((n, n_of_eqns))
        self.u[0, :] = self.y0

        for i in range(n - 1):
            self.i = i
            self.u[i + 1] = self.calculate()

        return self.u, self.t