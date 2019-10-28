import numpy as np


class Graph():
    def __init__(self, y0=0, x0=1, X=8, h=5):
        self.y0 = y0  # initial value for y
        self.x0 = x0  # initial value for x
        self.X = X  # final value for x
        self.h = h  # step

        self.exact_list = []  # exact solution
        self.euler_list = []  # euler method
        self.imp_euler_list = []  # improved euler method
        self.rk_list = []  # runge-kutta

        self.le_euler = []  # local error of euler method
        self.le_imp_euler = []  # local error of improved euler method
        self.le_rk = []  # local error of runge-kutta method

    def set_params(self, x0, y0, X, h):
        self.x0 = x0  # initial value for x
        self.y0 = y0  # initial value for y
        self.X = X  # final value for x
        self.h = h  # step

    # diff. eq. y' = y/x - xe^(y/x)
    def y_prime(self, x, y):
        return y / x - x * (pow(np.e, (y / x)))

    def exact_solution(self, x):
        return (-x) * np.log(x)

    def calc_exact(self):
        self.exact_list = []
        for i in np.arange(self.x0, self.X + self.h, self.h):
            self.exact_list.append(self.exact_solution(i))

        # Euler method y_n = y_n-1 + h*f(n-1, y_n-1)
        # Improved Euler method y_n = y_n-1 + h/2 * (f(n-1, y_n-1) + f(n, y_n-1 + h*f(n-1, y_n-1)))
        # Runge-Kutta method y_n =

    def init_lists(self):
        self.euler_list = []
        self.imp_euler_list = []
        self.rk_list = []
        self.le_euler = []
        self.le_imp_euler = []
        self.le_rk = []

        self.euler_list.append(self.y0)
        self.imp_euler_list.append(self.y0)
        self.rk_list.append(self.y0)

        self.le_euler.append(0)
        self.le_imp_euler.append(0)
        self.le_rk.append(0)

    def calculate(self):
        self.init_lists()

        self.calc_exact()
        self.calc_euler()
        self.calc_imp_euler()
        self.calc_rk()

    def calc_euler(self):
        for i in np.arange(self.x0 + self.h, self.X + self.h, self.h):
            e = self.euler_list[-1] + self.h * self.y_prime(i - self.h, self.euler_list[-1])

            exact = self.exact_solution(i)
            self.euler_list.append(e)
            self.le_euler.append(abs(exact - e))

    def calc_imp_euler(self):
        for i in np.arange(self.x0 + self.h, self.X + self.h, self.h):
            i_e = self.imp_euler_list[-1] + 0.5 * self.h * (self.y_prime(i - self.h, self.imp_euler_list[-1])
                    + self.y_prime(i - self.h, self.imp_euler_list[-1] + self.h * self.y_prime(i - self.h,
                                                           self.imp_euler_list[-1])))

            self.imp_euler_list.append(i_e)
            exact = self.exact_solution(i)
            self.le_imp_euler.append(abs(exact - i_e))

    def calc_rk(self):
        for i in np.arange(self.x0 + self.h, self.X + self.h, self.h):
            k1 = self.y_prime(i - self.h, self.rk_list[-1])
            k2 = self.y_prime(i - self.h / 2, self.rk_list[-1] + (self.h * k1) / 2)
            k3 = self.y_prime(i - self.h / 2, self.rk_list[-1] + (self.h * k2) / 2)
            k4 = self.y_prime(i, self.rk_list[-1] + self.h * k3)
            rk = self.rk_list[-1] + (self.h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            self.rk_list.append(rk)
            exact = self.exact_solution(i)
            self.le_rk.append(abs(exact - rk))

# Blue-le_euler, Orange-le_imp_euler, Green-le_rk
#
# df = pd.DataFrame(zip(ax, euler_list, imp_euler_list, rk_list, exact_list,
#                       [h for i in np.arange(x0, X + h, h)], le_euler, le_imp_euler, le_rk),
#                   columns=['x', 'euler', 'imp_euler', 'runge-kutta', 'exact', 'h',
#                            'le euler', 'le imp euler', 'le runge-kutta'])
