import numpy as np
from abc import ABC, abstractmethod


class AbstractExactSolution(ABC):
    @abstractmethod
    def calculate_exact(self):
        pass

    @abstractmethod
    def y_prime_ith(self, i, y):
        pass

    @abstractmethod
    def exact_ith(self, i):
        pass


class AbstractNumericalMethod(ABC):
    @abstractmethod
    def calculate_approximation(self):
        pass

    @abstractmethod
    def calculate_local_error(self):
        pass

    @abstractmethod
    def calculate_global_error(self):
        pass


class MyFunction(AbstractExactSolution):
    def __init__(self, x0=1, y0=0, x=8, n=5, n_start=1, n_end=5):
        # initial value for x
        self.x0 = x0
        # initial value for y
        self.y0 = y0
        # final value for interval (x0; x)
        self.x = x

        # Size of step
        self.h = float((self.x - self.x0) / n)

        # initial value for number of points
        self.n_start = n_start
        # final value for number of points
        self.n_end = n_end

        # coefficient in ivp
        self.c = self.x0 - pow(np.e, (-self.y0) / self.x0)

    def set_params(self, x0, y0, x, n, n_start, n_end):
        self.x0 = x0
        self.y0 = y0
        self.x = x

        self.h = float((self.x - self.x0) / n)
        self.n_start = n_start
        self.n_end = n_end

        self.axis = np.arange(self.x0, self.x, self.h)
        # self.step = 0.1 if self.h > 1 else self.h

        self.c = self.x0 - pow(np.e, (-self.y0) / self.x0)

    # diff. eq. y' = y/x - xe^(y/x)
    def y_prime_ith(self, i, y):
        return y / i - i * (pow(np.e, y / i))

    def exact_ith(self, i):
        # try:
        return (-i) * np.log(i - self.c)
        # except:
        # return (-x - self.h - 1) * np.log(x - self.h - self.c)

    def calculate_exact(self):
        exact_list = []

        for i in np.arange(self.x0, self.x, self.h):
            exact_list.append(self.exact_ith(i))

        return exact_list


class EulerApproximation(AbstractNumericalMethod):
    def __init__(self, graph):
        self.graph = graph

        self.e_axis = np.arange(self.graph.x0, self.graph.x, self.graph.h)
        self.g_axis = np.arange(self.graph.n_start, self.graph.n_end, 1)
        self.axis = np.arange(self.graph.x0, self.graph.x, self.graph.h)
        # self.step = 0.1 if self.h > 1 else self.h

    def update_axes(self):
        self.e_axis = np.arange(self.graph.x0, self.graph.x, self.graph.h)
        self.g_axis = np.arange(self.graph.n_start, self.graph.n_end, 1)
        self.axis = np.arange(self.graph.x0, self.graph.x, self.graph.h)
        # self.step = 0.1 if self.h > 1 else self.h

    # Euler method y_n = y_n-1 + h*f(n-1, y_n-1)
    def calculate_approximation(self):
        euler_list = [self.graph.y0]

        for i in np.arange(self.graph.x0 + self.graph.h, self.graph.x, self.graph.h):
            if i >= self.graph.x:
                break
            e = euler_list[-1] + self.graph.h * self.graph.y_prime_ith(i - self.graph.h, euler_list[-1])

            euler_list.append(e)

        return euler_list

    def calculate_local_error(self):
        le_euler = [0]
        for i in np.arange(self.graph.x0 + self.graph.h, self.graph.x, self.graph.h):
            if i >= self.graph.x:
                break

            e2 = self.graph.exact_ith(i - self.graph.h) + self.graph.h * self.graph.y_prime_ith(i - self.graph.h,
                                                                                                self.graph.exact_ith(
                                                                                                    i - self.graph.h))

            exact = self.graph.exact_ith(i)
            le_euler.append(abs(exact - e2))
        return le_euler

    def calculate_global_error(self):
        ge_euler = []

        for h in np.arange(self.graph.n_start, self.graph.n_end, 1):
            euler_list = [self.graph.y0]
            for i in np.arange(self.graph.x0 + h, self.graph.x, h):
                if i >= self.graph.x:
                    break
                e = euler_list[-1] + h * self.graph.y_prime_ith(i - h, euler_list[-1])

                euler_list.append(e)

            ge_euler.append(abs(self.graph.exact_ith(self.graph.x - h) - euler_list[-1]))

        return ge_euler


class ImpEulerApproximation(AbstractNumericalMethod):
    def __init__(self, graph):
        self.graph = graph

        self.e_axis = np.arange(self.graph.x0, self.graph.x, self.graph.h)
        self.g_axis = np.arange(self.graph.n_start, self.graph.n_end, 1)
        self.axis = np.arange(self.graph.x0, self.graph.x, self.graph.h)
        # self.step = 0.1 if self.h > 1 else self.h

    def update_axes(self):
        self.e_axis = np.arange(self.graph.x0, self.graph.x, self.graph.h)
        self.g_axis = np.arange(self.graph.n_start, self.graph.n_end, 1)
        self.axis = np.arange(self.graph.x0, self.graph.x, self.graph.h)
        # self.step = 0.1 if self.h > 1 else self.h

    # Improved Euler method y_n = y_n-1 + h/2 * (f(n-1, y_n-1) + f(n, y_n-1 + h*f(n-1, y_n-1)))
    def calculate_approximation(self):
        imp_euler_list = [self.graph.y0]

        for i in np.arange(self.graph.x0 + self.graph.h, self.graph.x, self.graph.h):
            if i >= self.graph.x:
                break
            prev = i - self.graph.h

            i_e = imp_euler_list[-1] + 0.5 * self.graph.h * (self.graph.y_prime_ith(prev, imp_euler_list[-1])
                                                             + self.graph.y_prime_ith(prev, imp_euler_list[-1] +
                                                                                      self.graph.h * self.graph.y_prime_ith(
                        prev, imp_euler_list[-1])))

            imp_euler_list.append(i_e)

        return imp_euler_list

    def calculate_local_error(self):
        le_imp_euler = [0]
        for i in np.arange(self.graph.x0 + self.graph.h, self.graph.x, self.graph.h):
            if i >= self.graph.x:
                break
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
        for h in np.arange(self.graph.n_start, self.graph.n_end, 1):
            imp_euler_list = [self.graph.y0]
            for i in np.arange(self.graph.x0 + h, self.graph.x, h):
                if i >= self.graph.x:
                    break
                prev = i - h
                i_e = imp_euler_list[-1] + 0.5 * h * (self.graph.y_prime_ith(prev, imp_euler_list[-1])
                                                      + self.graph.y_prime_ith(prev, imp_euler_list[-1] +
                                                                               h * self.graph.y_prime_ith(prev,
                                                                                                          imp_euler_list[
                                                                                                              -1])))

                imp_euler_list.append(i_e)

            ge_imp_euler.append(abs(self.graph.exact_ith(self.graph.x - h) - imp_euler_list[-1]))

        return ge_imp_euler


class RKApproximation(AbstractNumericalMethod):
    def __init__(self, graph):
        self.graph = graph

        self.e_axis = np.arange(self.graph.x0, self.graph.x, self.graph.h)
        self.g_axis = np.arange(self.graph.n_start, self.graph.n_end, 1)
        self.axis = np.arange(self.graph.x0, self.graph.x, self.graph.h)
        # self.step = 0.1 if self.h > 1 else self.h

    def update_axes(self):
        self.e_axis = np.arange(self.graph.x0, self.graph.x, self.graph.h)
        self.g_axis = np.arange(self.graph.n_start, self.graph.n_end, 1)
        self.axis = np.arange(self.graph.x0, self.graph.x, self.graph.h)
        # self.step = 0.1 if self.h > 1 else self.h

    def calculate_approximation(self):
        rk_list = [self.graph.y0]

        for i in np.arange(self.graph.x0 + self.graph.h, self.graph.x, self.graph.h):
            if i >= self.graph.x:
                break
            prev = i - self.graph.h

            k1 = self.graph.y_prime_ith(prev, rk_list[-1])
            k2 = self.graph.y_prime_ith(i - self.graph.h / 2, rk_list[-1] + (self.graph.h * k1) / 2)
            k3 = self.graph.y_prime_ith(i - self.graph.h / 2, rk_list[-1] + (self.graph.h * k2) / 2)
            k4 = self.graph.y_prime_ith(i, rk_list[-1] + self.graph.h * k3)
            rk = rk_list[-1] + (self.graph.h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            rk_list.append(rk)

        return rk_list

    def calculate_local_error(self):
        le_rk = [0]

        for i in np.arange(self.graph.x0 + self.graph.h, self.graph.x, self.graph.h):
            if i >= self.graph.x:
                break
            prev = i - self.graph.h

            k1_2 = self.graph.y_prime_ith(prev, self.graph.exact_ith(prev))
            k2_2 = self.graph.y_prime_ith(i - self.graph.h / 2, self.graph.exact_ith(prev) + (self.graph.h * k1_2) / 2)
            k3_2 = self.graph.y_prime_ith(i - self.graph.h / 2, self.graph.exact_ith(prev) + (self.graph.h * k2_2) / 2)
            k4_2 = self.graph.y_prime_ith(i, self.graph.exact_ith(prev) + self.graph.h * k3_2)
            rk2 = le_rk[-1] + (self.graph.h / 6) * (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2)

            exact = self.graph.exact_ith(i)
            le_rk.append(abs(exact - rk2))

        return le_rk

    def calculate_global_error(self):
        ge_rk = []

        for h in np.arange(self.graph.n_start, self.graph.n_end, 1):
            rk_list = [self.graph.y0]
            for i in np.arange(self.graph.x0 + h, self.graph.x, h):
                if i >= self.graph.x:
                    break
                prev = i - h

                k1 = self.graph.y_prime_ith(prev, rk_list[-1])
                k2 = self.graph.y_prime_ith(i - h / 2, rk_list[-1] + (h * k1) / 2)
                k3 = self.graph.y_prime_ith(i - h / 2, rk_list[-1] + (h * k2) / 2)
                k4 = self.graph.y_prime_ith(i, rk_list[-1] + h * k3)
                rk = rk_list[-1] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

                rk_list.append(rk)

            ge_rk.append(abs(self.graph.exact_ith(self.graph.x - h) - rk_list[-1]))

        return ge_rk
