import numpy as np


class Function:
    def __init__(self, y0=0, x0=1, X=8, N=5, h_start=1, h_end=5):
        self.y0 = y0  # initial value for y
        self.x0 = x0  # initial value for x
        self.X = X  # final value for x h
        self.h = float((self.X - self.x0) / N)  # Size of step
        self.h_start = h_start
        self.h_end = h_end

        self.c = self.x0 - pow(np.e, (-self.y0) / self.x0)

        self.axis = np.arange(self.x0, self.X, self.h)
        # self.step = 0.1 if self.h > 1 else self.h
        self.e_axis = np.arange(self.x0, self.X, self.h)
        self.g_axis = np.arange(self.h_start, self.h_end, (self.h_end - self.h_start) / 5)

    def set_params(self, x0, y0, X, N, h_start, h_end):
        self.x0 = x0  # initial value for x
        self.y0 = y0  # initial value for y
        self.X = X  # final value for x
        self.h = float((self.X - self.x0) / N)  # step
        self.h_start = h_start
        self.h_end = h_end

        self.axis = np.arange(self.x0, self.X, self.h)
        # self.step = 0.1 if self.h > 1 else self.h
        self.e_axis = np.arange(self.x0, self.X, self.h)
        self.g_axis = np.arange(self.h_start, self.h_end, (self.h_end - self.h_start) / 5)

        self.c = self.x0 - pow(np.e, (-self.y0) / self.x0)

    # diff. eq. y' = y/x - xe^(y/x)
    def y_prime(self, x, y):
        return y / x - x * (pow(np.e, (y / x)))

    def exact_solution(self, x):
        return (-x) * np.log(x - self.c)

    def calc_exact(self):
        exact_list = []

        for i in np.arange(self.x0, self.X, self.h):
            exact_list.append(self.exact_solution(i))

        return exact_list


class Graph(Function):
    # Euler method y_n = y_n-1 + h*f(n-1, y_n-1)
    def calc_euler(self):
        euler_list = [self.y0]
        le_euler = [0]

        for i in np.arange(self.x0 + self.h, self.X, self.h):
            if i >= self.X:
                break
            e = euler_list[-1] + self.h * self.y_prime(i - self.h, euler_list[-1])
            e2 = self.exact_solution(i - self.h) + self.h * self.y_prime(i - self.h, self.exact_solution(i - self.h))

            exact = self.exact_solution(i)

            euler_list.append(e)
            le_euler.append(abs(exact - e2))

        return euler_list, le_euler

    # Improved Euler method y_n = y_n-1 + h/2 * (f(n-1, y_n-1) + f(n, y_n-1 + h*f(n-1, y_n-1)))
    def calc_imp_euler(self):
        imp_euler_list = [self.y0]
        le_imp_euler = [0]

        for i in np.arange(self.x0 + self.h, self.X, self.h):
            if i >= self.X:
                break
            prev = i - self.h

            i_e = imp_euler_list[-1] + 0.5 * self.h * (self.y_prime(prev, imp_euler_list[-1])
                                                       + self.y_prime(prev, imp_euler_list[-1] +
                                                                      self.h * self.y_prime(prev,
                                                                                            imp_euler_list[-1])))

            i_e2 = self.exact_solution(prev) + 0.5 * self.h * (
                    self.y_prime(prev, self.exact_solution(prev))
                    + self.y_prime(prev, self.exact_solution(prev) +
                                   self.h * self.y_prime(prev, self.exact_solution(prev))))

            exact = self.exact_solution(i)

            imp_euler_list.append(i_e)
            le_imp_euler.append(abs(exact - i_e2))

        return imp_euler_list, le_imp_euler

    def calc_rk(self):
        rk_list = [self.y0]
        le_rk = [0]

        for i in np.arange(self.x0 + self.h, self.X, self.h):
            if i >= self.X:
                break
            prev = i - self.h

            k1 = self.y_prime(prev, rk_list[-1])
            k2 = self.y_prime(i - self.h / 2, rk_list[-1] + (self.h * k1) / 2)
            k3 = self.y_prime(i - self.h / 2, rk_list[-1] + (self.h * k2) / 2)
            k4 = self.y_prime(i, rk_list[-1] + self.h * k3)
            rk = rk_list[-1] + (self.h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            k1_2 = self.y_prime(prev, self.exact_solution(prev))
            k2_2 = self.y_prime(i - self.h / 2, self.exact_solution(prev) + (self.h * k1) / 2)
            k3_2 = self.y_prime(i - self.h / 2, self.exact_solution(prev) + (self.h * k2) / 2)
            k4_2 = self.y_prime(i, self.exact_solution(prev) + self.h * k3)
            rk2 = rk_list[-1] + (self.h / 6) * (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2)

            exact = self.exact_solution(i)
            rk_list.append(rk)
            le_rk.append(abs(exact - rk2))

        return rk_list, le_rk

    def calc_global_euler(self):
        ge_euler = []

        for h in np.arange(self.h_start, self.h_end, (self.h_end - self.h_start) / 5):
            euler_list = [self.y0]
            for i in np.arange(self.x0 + h, self.X, h):
                if i >= self.X:
                    break
                e = euler_list[-1] + h * self.y_prime(i - h, euler_list[-1])

                euler_list.append(e)

            ge_euler.append(abs(self.exact_solution(self.X-h) - euler_list[-1]))

        return ge_euler

    def calc_global_imp_euler(self):
        ge_imp_euler = []
        for h in np.arange(self.h_start, self.h_end, (self.h_end - self.h_start) / 5):
            imp_euler_list = [self.y0]
            for i in np.arange(self.x0 + h, self.X, h):
                if i >= self.X:
                    break
                prev = i - h
                i_e = imp_euler_list[-1] + 0.5 * h * (self.y_prime(prev, imp_euler_list[-1])
                                                      + self.y_prime(prev, imp_euler_list[-1] +
                                                                     h * self.y_prime(prev, imp_euler_list[-1])))

                imp_euler_list.append(i_e)

            ge_imp_euler.append(abs(self.exact_solution(self.X - h) - imp_euler_list[-1]))

        return ge_imp_euler

    def calc_global_rk(self):
        ge_rk = []

        for h in np.arange(self.h_start, self.h_end, (self.h_end - self.h_start) / 5):
            rk_list = [self.y0]
            for i in np.arange(self.x0 + h, self.X, h):
                if i >= self.X:
                    break
                prev = i - h

                k1 = self.y_prime(prev, rk_list[-1])
                k2 = self.y_prime(i - h / 2, rk_list[-1] + (h * k1) / 2)
                k3 = self.y_prime(i - h / 2, rk_list[-1] + (h * k2) / 2)
                k4 = self.y_prime(i, rk_list[-1] + h * k3)
                rk = rk_list[-1] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

                rk_list.append(rk)

            ge_rk.append(abs(self.exact_solution(self.X - h) - rk_list[-1]))

        return ge_rk
