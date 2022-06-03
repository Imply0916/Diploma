import autograd.numpy as np
from dataclasses import dataclass
from typing import Callable, Union


"""
ODE:
u' = f(u, t)
u(t_0) = U_t0
"""
@dataclass
class ODEModel:
    f: Callable[[np.array, np.array], np.array]  # problem function - f(u, t)
    exact_test_solution: Union[Callable[[np.array], np.array], None]  # function, when we know exact solution - u(t), None otherwise
    t0: np.array  # because we made general solve methods for arbitrary dimentions, then start point may be in 3-dim t0 = (2, 4, 5)
    T: np.array  # because we made general solve methods for arbitrary dimentions, then end point may be in 3-dim T = (6, 1, 0)
    y0: np.array  # value in t0 point, also for arbitrary dimentions
    number_of_points_to_discretization: int = 100


#####################################################################################################################################
# 1) Scalar Differential Equation

problem_scalar_1 = ODEModel(
    f = lambda t, y: -5. * y + t, 
    exact_test_solution = lambda t: np.exp(- 5. * t) + t / 5, 
    t0 = np.array([0]), 
    T = np.array([3]), 
    y0 = np.array([1])
)


problem__scalar_2 = ODEModel(
    f = lambda t, y: -5. * y, 
    exact_test_solution = lambda t: np.exp(- 5. * t), 
    t0 = np.array([0]), 
    T = np.array([3]), 
    y0 = np.array([1])
)


#####################################################################################################################################
# 2)  System of Differential Equations

#=====================
# 2.1) 


#=====================
#  3) Nonautonomous ODE
problem_nonatonomous_1 = ODEModel(
    f = lambda t, y: y * (1 - 2 * t), 
    exact_test_solution = lambda t: np.exp(t - t ** 2), 
    t0 = np.array([0.]), 
    T = np.array([2.]), 
    y0 = np.array([1.])
)
problem_nonatonomous_2 = ODEModel(
    f = lambda t, y: y * (5 - 7 * t), 
    exact_test_solution = None, 
    t0 = np.array([0.]), 
    T = np.array([2.]), 
    y0 = np.array([1.])
)




# bad solution:
problem_3 = ODEModel(
    f = lambda t, y: np.exp(2 * t) / np.exp(y), 
    exact_test_solution = lambda t: np.log10(np.exp(2 * t) / 2. + np.exp(4) / 2.), 
    t0 = np.array([2.]), 
    T = np.array([5.]), 
    y0 = np.array([4.])
)