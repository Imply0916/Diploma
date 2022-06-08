import autograd.numpy as np
from autograd import grad, jacobian
from scipy import linalg

from .ode_solver import ODESolver
from ode_models import ODEModel


class ExplicitRungeKutta(ODESolver):

    def __init__(self, ode_problem: ODEModel, A: np.array, b: np.array, c: np.array, tolerance: float):
        super().__init__(ode_problem, A, b, c, tolerance)
        self.h = self.t[1] - self.t[0]
        
    def phi(self, current_time, current_y):
        K = np.zeros(self.s, dtype=float)
        for s in range(self.s):
            t = current_time + self.c[s] * self.h
            y = current_y
            for j in range(s):
                y += self.A[s, j] * K[j] * self.h
            K[s] = self.f(t, y)
        
        return self.h * K.T @ self.b
        