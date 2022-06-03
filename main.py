import numpy as np
from matplotlib import pyplot as plt
from typing import Callable

import ode_solvers

from ode_models import *
from plot_tools import *
from butcher_tables import *



def solve(
    ode_problem: ODEModel,              # differential problem, which we want to solve,
    ode_solver: ode_solvers.ODESolver,  # ODE solver
    method: Callable,                   # Butcher matrix funciton
    tol: float                          # tolerance
):
    A, b, c = method()
    solver = ode_solver(ode_problem, A, b, c, tol)
    return solver.solve()

    


def solve_ode_test(
    ode_problem: ODEModel,          # differential problem, which we want to solve
    tol=1e-5                        # tolerance
):
    test_explicit_methods = [ForwardEuler, KuttaThirdOrderMethod]
    tests_implicit_methods = [GaussLegendreSixOrder, CrankNicolsonMethodSecondOrder]
    tests_diagonally_implicit_methods = [DIRKThirdOrder, DIRKFourOrder]
    
    test_methods = test_explicit_methods + tests_implicit_methods + tests_diagonally_implicit_methods

    # build exact solution points if exists:
    if ode_problem.exact_test_solution:
        time_points_exact = np.linspace(ode_problem.t0, ode_problem.T, ode_problem.number_of_points_to_discretization)
        exact_solution = ode_problem.exact_test_solution(time_points_exact)

    # create an figure to display plots
    figure, axes = generate_subplots(
        k=len(test_explicit_methods) + len(tests_implicit_methods) + len(tests_diagonally_implicit_methods),
        row_wise=True
    )

    noise = 0.01  # add some noise in order to look at solution when he very good)
    
    # EXPLISIT METHODS
    for k, method in enumerate(test_explicit_methods):
        u = solve(
            ode_problem=ode_problem,
            ode_solver=ode_solvers.ExplicitRungeKutta,
            method=method,
            tol=tol
        )
        # plot result:
        axes[k].plot(u[:,0], u[:,1] + noise, color='red', label=f"{method.__name__}")
        if ode_problem.exact_test_solution:
            axes[k].plot(time_points_exact, exact_solution, label="Exact solution")
        axes[k].grid(True)
        axes[k].set_title("Explicit Runge Kutta")
        axes[k].legend()

    # IMPLISIT METHODS
    for k, method in enumerate(tests_implicit_methods):
        k += len(test_explicit_methods)

        u = solve(
            ode_problem=ode_problem,
            ode_solver=ode_solvers.ImplicitRungeKutta,
            method=method,
            tol=tol
        )
        # plot result:
        axes[k].plot(u[:,0], u[:,1] + noise, color='red', label=f"{method.__name__}")
        if ode_problem.exact_test_solution:
            axes[k].plot(time_points_exact, exact_solution, label="Exact solution")
        axes[k].grid(True)
        axes[k].set_title("Implicit Runge Kutta")
        axes[k].legend()
    
    # DIAGONALY IMPLISIT METHODS
    for k, method in enumerate(tests_diagonally_implicit_methods):
        k += len(test_explicit_methods + tests_implicit_methods)

        u = solve(
            ode_problem=ode_problem,
            ode_solver=ode_solvers.DiagonallyImplicitRungeKutta,
            method=method,
            tol=tol
        )
        # plot result:
        axes[k].plot(u[:,0], u[:,1] + noise, color='red', label=f"{method.__name__}")
        if ode_problem.exact_test_solution:
            axes[k].plot(time_points_exact, exact_solution, label="Exact solution")
        axes[k].grid(True)
        axes[k].set_title("Diagonally Implicit Runge Kutta")
        axes[k].legend()
    figure.canvas.set_window_title("Solution for ode problem")
    plt.show()


def example_1():
    problems = [
        problem_scalar_1, 
        problem__scalar_2, 
        problem_nonatonomous_1,
    ]

    # solve problems all avalible methods:
    for problem in problems:
        solve_ode_test(ode_problem=problem)


def example_2():
    # without exact solution
    solve_ode_test(ode_problem=problem_nonatonomous_2)


if __name__ == "__main__":
    example_1()
    example_2()
    