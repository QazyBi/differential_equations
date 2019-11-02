import numpy as np


class Function:
    def __init__(self, y0=0, x0=1, X=8, h=5):
        self.y0 = y0  # initial value for y
        self.x0 = x0  # initial value for x
        self.X = X  # final value for x h
        self.h = h  # step

        self.axis = np.arange(self.x0, self.X, self.h)
        self.step = 0.1 if self.h > 1 else self.h
        self.e_axis = np.arange(self.x0, self.X, self.step)

    def set_params(self, x0, y0, X, h):
        self.x0 = x0  # initial value for x
        self.y0 = y0  # initial value for y
        self.X = X  # final value for x
        self.h = h  # step
        self.axis = np.arange(self.x0, self.X, self.h)
        self.step = 0.1 if self.h > 1 else self.h
        self.e_axis = np.arange(self.x0, self.X, self.step)

    # diff. eq. y' = y/x - xe^(y/x)
    def y_prime(self, x, y):
        return y / x - x * (pow(np.e, (y / x)))

    def exact_solution(self, x):
        return (-x) * np.log(x)

    def calc_exact(self):
        exact_list = []
        for i in np.arange(self.x0, self.X, self.step):
            exact_list.append(self.exact_solution(i))

        return exact_list


class Graph(Function):
    # Euler method y_n = y_n-1 + h*f(n-1, y_n-1)
    def calc_euler(self):
        euler_list = []
        le_euler = []
        euler_list.append(self.y0)
        le_euler.append(0)

        for i in np.arange(self.x0 + self.h, self.X, self.h):
            e = euler_list[-1] + self.h * self.y_prime(i - self.h, euler_list[-1])

            exact = self.exact_solution(i)
            euler_list.append(e)
            le_euler.append(abs(exact - e))

        return euler_list, le_euler

    # Improved Euler method y_n = y_n-1 + h/2 * (f(n-1, y_n-1) + f(n, y_n-1 + h*f(n-1, y_n-1)))
    def calc_imp_euler(self):
        imp_euler_list = []
        le_imp_euler = []
        imp_euler_list.append(self.y0)
        le_imp_euler.append(0)

        for i in np.arange(self.x0 + self.h, self.X, self.h):
            i_e = imp_euler_list[-1] + 0.5 * self.h * (self.y_prime(i - self.h, imp_euler_list[-1])
                                                       + self.y_prime(i - self.h, imp_euler_list[-1] +
                                                                      self.h * self.y_prime(i - self.h,
                                                                                            imp_euler_list[-1])))

            imp_euler_list.append(i_e)
            exact = self.exact_solution(i)
            le_imp_euler.append(abs(exact - i_e))

        return imp_euler_list, le_imp_euler

    def calc_rk(self):
        rk_list = []
        le_rk = []
        rk_list.append(self.y0)
        le_rk.append(0)

        for i in np.arange(self.x0 + self.h, self.X, self.h):
            k1 = self.y_prime(i - self.h, rk_list[-1])
            k2 = self.y_prime(i - self.h / 2, rk_list[-1] + (self.h * k1) / 2)
            k3 = self.y_prime(i - self.h / 2, rk_list[-1] + (self.h * k2) / 2)
            k4 = self.y_prime(i, rk_list[-1] + self.h * k3)
            rk = rk_list[-1] + (self.h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            rk_list.append(rk)
            exact = self.exact_solution(i)
            le_rk.append(abs(exact - rk))

        return rk_list, le_rk
