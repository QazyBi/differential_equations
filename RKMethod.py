import numpy as np
from AbstractNumericalMethod import AbstractNumericalMethod


class RKApproximation(AbstractNumericalMethod):
    def __init__(self, graph, n_start=2, n_end=5):
        self.graph = graph
        self.color = 'magenta'
        self.label = 'Runge-Kutta'

        self.n_start = n_start
        self.n_end = n_end

    def set_params(self, graph, color, label, n_start=2, n_end=5):
        if graph is not None:
            self.graph = graph
        if color is not None:
            self.color = color
        if label is not None:
            self.label = label

        self.n_start = n_start
        self.n_end = n_end

    def axis_global_error(self):
        return np.arange(self.n_start, self.n_end + 1, 1)
        # self.step = 0.1 if self.h > 1 else self.h

    def axis(self, step):
        return [round(x, 4) for x in np.arange(self.graph.x0, self.graph.x, step) if
                round(x, 4) <= self.graph.x_limit + 0.00001]

    def calculate_approximation(self, step):
        rk_list = [self.graph.y0]

        for i in self.axis(step):
            if i == self.graph.x0:
                continue
            prev = i - step

            k1 = self.graph.y_prime_ith(prev, rk_list[-1])
            k2 = self.graph.y_prime_ith(i - step / 2, rk_list[-1] + (step * k1) / 2)
            k3 = self.graph.y_prime_ith(i - step / 2, rk_list[-1] + (step * k2) / 2)
            k4 = self.graph.y_prime_ith(i, rk_list[-1] + step * k3)
            rk = rk_list[-1] + (step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            rk_list.append(rk)

        return rk_list

    def calculate_local_error(self):
        le_rk = [0]

        for i in self.axis(self.graph.h):
            if i == self.graph.x0:
                continue
            prev = i - self.graph.h

            k1_2 = self.graph.y_prime_ith(prev, self.graph.exact_ith(prev))
            k2_2 = self.graph.y_prime_ith(i - self.graph.h / 2, self.graph.exact_ith(prev) + (self.graph.h * k1_2) / 2)
            k3_2 = self.graph.y_prime_ith(i - self.graph.h / 2, self.graph.exact_ith(prev) + (self.graph.h * k2_2) / 2)
            k4_2 = self.graph.y_prime_ith(i, self.graph.exact_ith(prev) + self.graph.h * k3_2)
            rk2 = self.graph.exact_ith(prev) + (self.graph.h / 6) * (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2)

            exact = self.graph.exact_ith(i)
            le_rk.append(abs(exact - rk2))

        return le_rk

    def calculate_global_error(self):
        ge_rk = []
        for n in self.axis_global_error():
            step = float((self.graph.x_limit - self.graph.x0) / n)
            ge_rk.append(abs(self.graph.exact_ith(self.graph.x_limit) - self.calculate_approximation(step)[-1]))

        return ge_rk
