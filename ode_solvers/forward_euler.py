from .ode_solver import ODESolver



class ForwardEuler(ODESolver):
    b = []
    A = []
    c = []
    
    def calculate(self):
        self.b = []
        u, f, i, t = self.u, self.f, self.i, self.t
        dt = t[i + 1] - t[i]
        return u[i, :] + dt * f(t[i], u[i, :])

    def phi(self, current_time_step, y0):
        pass