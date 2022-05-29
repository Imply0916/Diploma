from .ode_solver import ODESolver


class RungeKutta4(ODESolver):
    b = []
    A = []
    c = []

    def calculate(self):
        u, f, i, t = self.u, self.f, self.i, self.t
        dt = t[i + 1] - t[i]
        dt2 = dt / 2.
        K1 = dt * f(t[i], u[i, :])
        K2 = dt * f(t[i] + dt2, u[i, :] + 0.5 * K1)
        K3 = dt * f(t[i] + dt2, u[i, :] + 0.5 * K2)
        K4 = dt * f(t[i] + dt, u[i, :] + K3)
        return u[i, :] + (1 / 6) * (K1 + 2 * K2 + 2 * K3 + K4)

    def phi(self, current_time_step, y0):
        pass

