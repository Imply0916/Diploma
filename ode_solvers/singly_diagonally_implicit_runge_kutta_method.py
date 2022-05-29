# implementation of Singly Diagonally Implicit Runge–Kutta Method(SDIRK)

import autograd.numpy as np
from autograd import grad, jacobian
from scipy import linalg

from .implicit_runge_kutta import ImplicitRungeKutta


class SinglyDiagonallyImplicitRungeKuttaMethod(ImplicitRungeKutta):

    def phi_solve(self, t0, y0, init_val, J, M):
        """
        This function solves F(Y_i)=0 by solving s systems of size m 
            x m each.
        Newton’s method is used with an initial guess init_val.

        Parameters:
        -------------
        t0 = float, current timestep
        y0 = 1 x m vector, the last solution y_n. Where m is the length of the initial condition y_0 of the IVP.
        init_val = initial guess for the Newton iteration
        J = m x m matrix, the Jacobian matrix of f() evaluated in y_i
        M = maximal number of Newton iterations

        Returns:
        -------------
        The stage derivative Y’_i
        """
        JJ = np.eye(self.m) - self.h * self.A[0,0] * J
        lu_factor = linalg.lu_factor(JJ)
        for i in range(M):
            init_val, norm_d = self.phi_newtonstep(t0, y0, init_val, J, lu_factor)
            if norm_d < self.tol:
                break
            elif i == M - 1:
                raise ValueError("The Newton iteration did noconverge.")
        return init_val

    

    def phi_newtonstep(self, t0, y0, init_val, J, lu_factor):
        """
        Takes one Newton step by solvning
        G’(Y_i)(Y^(n+1)_i-Y^(n)_i)=-G(Y_i)
        where G(Y_i) = Y_i - haY’_i - y_n - h*sum(a_{ij}*Y’_j) for j=1,...,i-1

        Parameters:
        -------------
        t0 = float, current timestep
        y0 = 1 x m vector, the last solution y_n. Where m is the length of the initial condition y_0 of the IVP.
        init_val = initial guess for the Newton iteration
        lu_factor = (lu, piv) see documentation for linalg.lu_factor

        Returns:
        The difference Y^(n+1)_i-Y^(n)_i
        """
        x = []
        for i in range(self.s): # solving the s mxm systems
            rhs = -self.F(
                init_val.flatten(), t0, y0
            )[i * self.m : (i + 1) * self.m] + np.sum(
                [self.h * self.A[i,j] * J @ x[j] for j in range(i)],
                axis = 0
            )
            d = linalg.lu_solve(lu_factor, rhs)
            x.append(d)
        return init_val + x, linalg.norm(x)
