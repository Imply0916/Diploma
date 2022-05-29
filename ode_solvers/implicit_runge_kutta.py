import autograd.numpy as np
from autograd import grad, jacobian
from scipy import linalg

from .ode_solver import ODESolver


class ImplicitRungeKutta(ODESolver):

    def phi(self, t0, y0):
        """
        Calculates the summation of b_j*Y_j in one step of the RungeKutta method with
        y_{n+1} = y_{n} + h * sum_{j=1}^{s} b_{j}*Y
        where j=1,2,...,s, and s is the number of stages, b the nodes, and Y the stage values of the method.
        Parameters:
        -------------
        t0 = float, current timestep
        y0 = 1 x m vector, the last solution y_n. Where m is the length of the initial condition y_0 of the IVP.
        """
        M = 1000 # max number of newton iterations

        stage_der = np.array(self.s * [self.f(t0, y0)]) # initial value: Y’_0
        J = jacobian(self.f)(t0, y0)
        stage_val = self.phi_solve(t0, y0, stage_der, J, M)

        return np.array([
            self.b @ stage_val.reshape(self.s,self.m)[:,j] for j in range(self.m)
        ])

    def phi_solve(self, t0, y0, init_val, J, M):
        """
        This function solves the sm x sm system F(Y_i)=0 by Newton’s method with an initial guess init_val.
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

        JJ = np.eye(self.s * self.m) - self.h * np.kron(self.A, J)
        lu_factor = linalg.lu_factor(JJ)
        for i in range(M):
            init_val, norm_d = self.phi_newtonstep(t0, y0, init_val, lu_factor)
            if norm_d < self.tol:
                break
            elif i == M - 1:
                raise ValueError("The Newton iteration did not converge.")
        return init_val


    def phi_newtonstep(self, t0, y0, init_val, lu_factor):
        """
        Takes one Newton step by solvning
        G’(Y_i)(Y^(n+1)_i-Y^(n)_i) = -G(Y_i), where
        G(Y_i) = Y_i - y_n - h*sum(a_{ij}*Y’_j) for j = 1,...,s
        Parameters:
        -------------
        t0 = float, current timestep
        y0 = 1 x m vector, the last solution y_n. Where m is the length of the initial condition y_0 of the IVP.
        init_val = initial guess for the Newton iteration
        lu_factor = (lu, piv) see documentation for linalg.lu_factor
        Returns:
        The difference Y^(n+1)_i-Y^(n)_i
        """
        d = linalg.lu_solve(lu_factor, -self.F(init_val.flatten(), t0, y0))
        return init_val.flatten() + d, linalg.norm(d)
    
    def F(self, stage_der, t0, y0):
        """
        Returns the subtraction Y’_{i}-f(t_{n}+c_{i}*h, Y_{i}), where Y are
        the stage values, Y’ the stage derivatives and f the function of
        the IVP y’=f(t,y) that should be solved by the RK-method.
        Parameters:
        -------------
        stage_der = initial guess of the stage derivatives Y’
        t0 = float, current timestep
        y0 = 1 x m vector, the last solution y_n. Where m is the length of the initial condition y_0 of the IVP.
        """
        stage_der_new = np.empty((self.s,self.m)) # the i:th stage_der is on the i:th row
        for i in range(self.s): # iterate over all stage_der
            stageVal = y0 + np.array([
                self.h * np.dot(self.A[i,:],
                stage_der.reshape(self.s, self.m)[:, j]) for j in range(self.m)
            ])
            stage_der_new[i, :] = self.f(t0 + self.c[i] * self.h, stageVal) # the ith stage_der is set on the ith row
        return stage_der - stage_der_new.reshape(-1)
