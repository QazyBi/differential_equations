import numpy as np
from abc import ABC, abstractmethod


class AbstractExactSolution(ABC):
    @abstractmethod
    def calculate_exact(self):
        pass

    @abstractmethod
    def y_prime_ith(self, i: float, y: float):
        pass

    @abstractmethod
    def exact_ith(self, i: float):
        pass


class AbstractNumericalMethod(ABC):
    @abstractmethod
    def calculate_approximation(self, step):
        pass

    @abstractmethod
    def calculate_local_error(self):
        pass

    @abstractmethod
    def calculate_global_error(self):
        pass


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


class EulerApproximation(AbstractNumericalMethod):
    def __init__(self, graph):
        self.graph = graph
        self.color = '#F6D349'
        self.label = 'Euler'
        self.g_axis = np.arange(self.graph.n_start, self.graph.n_end + 1, 1)
        # self.step = 0.1 if self.h > 1 else self.h

    def axis(self, step):
        return [round(x, 2) for x in np.arange(self.graph.x0, self.graph.x, step) if round(x, 3) <= self.graph.x_limit]

    def update_axes(self):
        self.g_axis = np.arange(self.graph.n_start, self.graph.n_end + 1, 1)
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
            le_euler.append(abs(exact - e2))
        return le_euler

    def calculate_global_error(self):
        ge_euler = []

        for n in self.g_axis:
            step = float((self.graph.x_limit - self.graph.x0) / n)
            ge_euler.append(abs(self.graph.exact_ith(self.graph.x_limit) - self.calculate_approximation(step)[-1]))
        return ge_euler


class ImpEulerApproximation(AbstractNumericalMethod):
    def __init__(self, graph):
        self.graph = graph
        self.color = '#8FDDD3'
        self.label = 'Improved Euler'
        self.g_axis = np.arange(self.graph.n_start, self.graph.n_end + 1, 1)
        # self.step = 0.1 if self.h > 1 else self.h

    def update_axes(self):
        self.g_axis = np.arange(self.graph.n_start, self.graph.n_end + 1, 1)
        # self.step = 0.1 if self.h > 1 else self.h

    def axis(self, step):
        return [round(x, 4) for x in np.arange(self.graph.x0, self.graph.x, step) if round(x, 4) <= self.graph.x_limit]

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
        for n in self.g_axis:
            step = float((self.graph.x_limit - self.graph.x0) / n)
            ge_imp_euler.append(abs(self.graph.exact_ith(self.graph.x_limit) - self.calculate_approximation(step)[-1]))

        return ge_imp_euler


class RKApproximation(AbstractNumericalMethod):
    def __init__(self, graph):
        self.graph = graph
        self.color = 'magenta'
        self.label = 'Runge-Kutta'
        self.g_axis = np.arange(self.graph.n_start, self.graph.n_end + 1, 1)
        # self.step = 0.1 if self.h > 1 else self.h

    def update_axes(self):
        self.g_axis = np.arange(self.graph.n_start, self.graph.n_end + 1, 1)
        # self.step = 0.1 if self.h > 1 else self.h

    def axis(self, step):
        return [round(x, 4) for x in np.arange(self.graph.x0, self.graph.x, step) if round(x, 4) <= self.graph.x_limit+0.00001]

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
        for n in self.g_axis:
            step = float((self.graph.x_limit - self.graph.x0) / n)
            ge_rk.append(abs(self.graph.exact_ith(self.graph.x_limit) - self.calculate_approximation(step)[-1]))

        return ge_rk
