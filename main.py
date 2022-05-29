import numpy as np
from matplotlib import pyplot as plt

from ode_models import *
from plot_tools import *
from ode_solvers import *



# 72 page in (Solving Ordinary Differential Equations II Stiff and Differential-Algebraic Problems (Ernst Hairer, Gerhard Wanner (auth.)))
class Gauss(ImplicitRungeKutta):  # order 6
    A = np.array([
        [5 / 36, 2 / 9 - np.sqrt(15) / 15, 5 / 36 - np.sqrt(15) / 30],
        [ 5 / 36 + np.sqrt(15) / 24, 2 / 9, 5 / 36 - np.sqrt(15) / 24],
        [ 5 / 36 + np.sqrt(15) / 30, 2 / 9 + np.sqrt(15) / 15, 5 / 36]
    ])
    b = np.array([5 / 18, 4 / 9, 5 / 18])
    c = np.array([1 / 2 - np.sqrt(15) / 10, 1 / 2, 1 / 2 + np.sqrt(15) / 10])

# 107 page in (Computer Methods for Ordinary Differential Equations and Differential-Algebraic Equations (Uri M. Ascher, Linda Ruth Petzold))
class SDIRK_tableau2s(SinglyDiagonallyImplicitRungeKuttaMethod):  # order 3
    p = (3 - np.sqrt(3)) / 6
    A = np.array([[p, 0], [1 - 2 * p, p]])
    b = np.array([1 / 2, 1 / 2])
    c = np.array([p, 1 - p])


# 107 page in (Solving Ordinary Differential Equations II Stiff and Differential-Algebraic Problems (Ernst Hairer, Gerhard Wanner (auth.)))
class SDIRK_tableau5s(SinglyDiagonallyImplicitRungeKuttaMethod):  # order 4
    A = np.array([
        [1 / 4, 0, 0, 0, 0], 
        [1 / 2, 1 / 4, 0, 0, 0], 
        [17 / 50, -1 / 25, 1 / 4, 0, 0],
        [371 / 1360, -137 / 2720, 15 / 544, 1 / 4, 0],
        [25 / 24, -49 / 48, 125 / 16, -85 / 12, 1 / 4]
    ])
    b = np.array([25 / 24, -49 / 48, 125 / 16, -85 / 12, 1 / 4])
    c = np.array([1 / 4, 3 / 4, 11 / 20, 1 / 2, 1])



class Test(ImplicitRungeKutta):
    A = np.array([
        [0.1129994793231561, -0.040309220723521944, 0.025802377420336385, -0.009904676507266638], 
        [0.2343839957474003, 0.20689257393535793, -0.04785712804854046, 0.016047422806516585], 
        [0.21668178462325022, 0.40612326386737374, 0.18903651817006234, -0.024182104899835286], 
        [0.220462211176768, 0.388193468843174, 0.328844319980063, 0.0624999999999960]
    ])
    b = np.array([0.220462211176768, 0.388193468843174, 0.328844319980063, 0.0624999999999960])
    c = np.array([0.08858795951270391, 0.40946686444073427, 0.787659461760851, 1.0])



### tests:
 
def solve_ode_test(ode_problem: ODEModel):
    # set tolerance
    tol = 1e-5

    # build exact solution points:
    if ode_problem.exact_test_solution:
        time_points_exact = np.linspace(ode_problem.t0, ode_problem.T, ode_problem.number_of_points_to_discretization)
        exact_solution = ode_problem.exact_test_solution(time_points_exact)
    
    # set tested methods
    tests_1 = [(Gauss, 6, "o"), (SDIRK_tableau2s, 3, "v"), (SDIRK_tableau5s, 4, "s"), (Test, 15)]
    tests_2 = [(ForwardEuler, 1), (RungeKutta4, 4)]

    # create an figure to display plots
    figure, axes = generate_subplots(len(tests_1 + tests_2), row_wise=True)

    noise = 0.01 # add some noise in order to look at solution when he very good)

    current_method_num = 0
    for k, method in enumerate(tests_1):
        # solve by selected method
        solver = method[0](ode_problem, tol)
        u = solver.solve()
        # plot result:
        axes[k].plot(u[:,0], u[:,1] + noise, color='red', label=f"{method[0].__name__}, order={method[1]}")
        if ode_problem.exact_test_solution:
            axes[k].plot(time_points_exact, exact_solution, label="Exact solution")
        axes[k].grid(True)
        axes[k].legend()

    for k, method in enumerate(tests_2):
        k += len(tests_1)
        # solve by selected method
        solver = method[0](ode_problem, tol)
        u, t = solver.solve_()
        # plot result:
        axes[k].plot(t, u[:, 0], color='red', label=f"{method[0].__name__}, order={method[1]}")
        if ode_problem.exact_test_solution:
            axes[k].plot(time_points_exact, exact_solution, label="Exact solution")
        axes[k].grid(True)
        axes[k].legend()
   
    plt.show()



if __name__ == "__main__":
    solve_ode_test(problem_scalar_1)
    solve_ode_test(problem__scalar_2)
    solve_ode_test(problem_nonatonomous_1)
    # solve_ode_test(problem_nonlinear_1)
    # solve_ode_test(problem_2)
    # solve_ode_test(problem_3)
    # solve_ode_test(problem_4)

    

