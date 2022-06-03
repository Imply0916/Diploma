import autograd.numpy as np
from autograd import grad, jacobian
from scipy import linalg
from typing import Callable
import numpy.typing as npt

from ode_models import ODEModel


class ODESolver:
    """ODESolver superclass

        ODE:
        u' = f(u, t)
        u(t_0) = U_t0
    """

    def __init__(self, ode_problem: ODEModel, A: np.array, b: np.array, c: np.array, tolerance: float):
        self.f = ode_problem.f
        self.y0 = ode_problem.y0.astype(float)  # initial condition
        self.num_init_conditions = len(self.y0)

        self.u = None  # solution
        self.i = None  # current number of step iteration

        self.h = (ode_problem.T - ode_problem.t0) / (ode_problem.number_of_points_to_discretization + 1)
        self.t = np.linspace(ode_problem.t0, ode_problem.T, ode_problem.number_of_points_to_discretization + 2)  # array of time points corresponding to solution

        self.tol = tolerance

        # setting Butcher table properties:
        self.A = A
        self.b = b
        self.c = c
        self.s = len(self.b)
        
    def step(self):
        ti, yi = self.t[0], self.y0  # initial condition points
        current_time_point = ti
        yield ti , np.array(yi)  # first point (begging point)
        for ti in self.t[1:]:
            yi += self.h * self.phi(current_time_point, yi)
            current_time_point = ti
            yield ti, np.array(yi)

    def solve(self):
        return np.array(list(self.step()))

    def phi(self, current_time, current_y):
        """Advance solution one time step."""
        raise NotImplementedError
    