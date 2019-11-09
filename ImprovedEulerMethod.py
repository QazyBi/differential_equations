import numpy as np
from AbstractNumericalMethod import AbstractNumericalMethod


class ImpEulerApproximation(AbstractNumericalMethod):
    def __init__(self, graph, n_start=2, n_end=5):
        self.graph = graph
        self.color = '#8FDDD3'
        self.label = 'Improved Euler'

        self.n_start = n_start
        self.n_end = n_end

    def axis_global_error(self):
        return np.arange(self.n_start, self.n_end + 1, 1)

    def axis(self, step):
        return [round(x, 4) for x in np.arange(self.graph.x0, self.graph.x, step) if round(x, 4) <= self.graph.x_limit]

    def set_params(self, graph, color, label, n_start=2, n_end=5):
        if graph is not None:
            self.graph = graph
        if color is not None:
            self.color = color
        if label is not None:
            self.label = label

        self.n_start = n_start
        self.n_end = n_end

    # Improved Euler method y_n = y_n-1 + h/2 * (f(n-1, y_n-1) + f(n, y_n-1 + h*f(n-1, y_n-1)))
    def calculate_approximation(self, step):
        imp_euler_list = [self.graph.y0]

        for i in self.axis(step):
            if i == self.graph.x0:
                continue
            prev = i - step

            i_e = imp_euler_list[-1] + 0.5 * step * (self.graph.y_prime_ith(prev, imp_euler_list[-1])
                                                     + self.graph.y_prime_ith(prev, imp_euler_list[-1] +
                                                                              step * self.graph.y_prime_ith(prev,
                                                                                                            imp_euler_list[
                                                                                                                -1])))

            imp_euler_list.append(i_e)
        return imp_euler_list

    def calculate_local_error(self):
        le_imp_euler = [0]
        for i in self.axis(self.graph.h):
            if i == self.graph.x0:
                continue
            prev = i - self.graph.h
            i_e2 = self.graph.exact_ith(prev) + 0.5 * self.graph.h * (
                    self.graph.y_prime_ith(prev, self.graph.exact_ith(prev))
                    + self.graph.y_prime_ith(prev, self.graph.exact_ith(prev) +
                                             self.graph.h * self.graph.y_prime_ith(prev, self.graph.exact_ith(prev))))

            exact = self.graph.exact_ith(i)
            le_imp_euler.append(abs(exact - i_e2))

        return le_imp_euler

    def calculate_global_error(self):
        ge_imp_euler = []
        for n in self.axis_global_error():
            step = float((self.graph.x_limit - self.graph.x0) / n)
            ge_imp_euler.append(abs(self.graph.exact_ith(self.graph.x_limit) - self.calculate_approximation(step)[-1]))

        return ge_imp_euler
