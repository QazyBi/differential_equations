import numpy as np
from AbstractNumericalMethod import AbstractNumericalMethod


class EulerApproximation(AbstractNumericalMethod):
    def __init__(self, graph, n_start=2, n_end=5):
        self.graph = graph
        self.color = '#F6D349'
        self.label = 'Euler'

        # initial value for number of points
        self.n_start = n_start
        # final value for number of points
        self.n_end = n_end

        # self.step = 0.1 if self.h > 1 else self.h
    def set_params(self, graph, color, label, n_start=2, n_end=5):
        if graph is not None:
            self.graph = graph
        if color is not None:
            self.color = color
        if label is not None:
            self.label = label

        self.n_start = n_start
        self.n_end = n_end

    def axis(self, step):
        return [round(x, 2) for x in np.arange(self.graph.x0, self.graph.x, step) if round(x, 3) <= self.graph.x_limit]

    def axis_global_error(self):
        return np.arange(self.n_start, self.n_end + 1, 1)
        # self.step = 0.1 if self.h > 1 else self.h

    # Euler method y_n = y_n-1 + h*f(n-1, y_n-1)
    def calculate_approximation(self, step):
        euler_list = [self.graph.y0]

        for i in self.axis(step):
            if i == self.graph.x0:
                continue
            e = euler_list[-1] + step * self.graph.y_prime_ith(i - step, euler_list[-1])
            euler_list.append(e)

        return euler_list

    def calculate_local_error(self):
        le_euler = [0]
        for i in self.axis(self.graph.h):
            if i == self.graph.x0:
                continue
            e2 = self.graph.exact_ith(i - self.graph.h) + self.graph.h * self.graph.y_prime_ith(i - self.graph.h,
                                                                                                self.graph.exact_ith(
                                                                                                    i - self.graph.h))

            exact = self.graph.exact_ith(i)
            # print("i:{} e2:{} le:{}".format(i, e2, abs(exact - e2)))
            le_euler.append(abs(exact - e2))

        return le_euler

    def calculate_global_error(self):
        ge_euler = []

        for n in self.axis_global_error():
            step = float((self.graph.x_limit - self.graph.x0) / n)
            ge_euler.append(abs(self.graph.exact_ith(self.graph.x_limit) - self.calculate_approximation(step)[-1]))
        return ge_euler
