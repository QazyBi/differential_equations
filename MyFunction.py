import numpy as np
from AbstractExactSolution import AbstractExactSolution


class MyFunction(AbstractExactSolution):
    def __init__(self, x0=1, y0=0, x=8, n=4, n_start=1, n_end=5):
        # initial value for x
        self.x0 = x0
        # initial value for y
        self.y0 = y0

        # Size of step
        self.h = float((x - x0) / n)

        # final value for interval (x0; x)
        self.x = round(x + self.h, 3)
        self.x_limit = x
        # initial value for number of points
        self.n_start = n_start
        # final value for number of points
        self.n_end = n_end

        self.e_axis = np.arange(self.x0, self.x, self.h)
        # coefficient in ivp
        self.c = self.x0 - pow(np.e, (-self.y0) / self.x0)

    def set_params(self, x0, y0, x, n, n_start, n_end):
        self.x0 = x0
        self.y0 = y0

        # Size of step
        self.h = float((x - x0) / n)

        # final value for interval (x0; x)
        self.x_limit = x
        self.x = round(x + self.h, 3)
        self.n_start = n_start
        self.n_end = n_end

        self.e_axis = [round(x, 2) for x in np.arange(self.x0, self.x, self.h) if round(x, 3) <= self.x_limit]
        # self.step = 0.1 if self.h > 1 else self.h

        self.c = self.x0 - pow(np.e, (-self.y0) / self.x0)

    # diff. eq. y' = y/x - xe^(y/x)
    def y_prime_ith(self, i, y):
        return y / i - i * (pow(np.e, y / i))

    def exact_ith(self, i) -> float:
        return (-i) * np.log(i - self.c)

    def calculate_exact(self) -> list:
        exact_list = []

        for i in self.e_axis:
            exact_list.append(self.exact_ith(i))
        return exact_list
