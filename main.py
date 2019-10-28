  import sys
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QGridLayout

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt


class Window(QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # a figure instance to plot on
        self.figure_euler = plt.figure()
        self.figure_imp_euler = plt.figure()
        self.figure_rk = plt.figure()
        self.figure_exact = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas_euler = FigureCanvas(self.figure_euler)
        self.canvas_imp_euler = FigureCanvas(self.figure_imp_euler)
        self.canvas_rk = FigureCanvas(self.figure_rk)
        self.canvas_exact = FigureCanvas(self.figure_exact)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas_euler, self)

        # Just some button connected to `plot` method
        self.b_plot = QPushButton('Plot ')
        self.b_plot.clicked.connect(self.plot)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)

        glayout = QGridLayout()

        glayout.addWidget(self.canvas_euler, 0, 0)
        glayout.addWidget(self.canvas_imp_euler, 0, 1)
        glayout.addWidget(self.canvas_rk, 1, 0)
        glayout.addWidget(self.canvas_exact, 1, 1)

        layout.addLayout(glayout)
        layout.addWidget(self.b_plot)
        self.setLayout(layout)

    def plot(self):
        self.figure_euler.clear()
        self.figure_imp_euler.clear()
        self.figure_rk.clear()
        self.figure_exact.clear()

        # create an axis
        ax_euler = self.figure_euler.add_subplot(111)
        ax_imp_euler = self.figure_imp_euler.add_subplot(111)
        ax_rk = self.figure_rk.add_subplot(111)
        ax_exact = self.figure_exact.add_subplot(111)

        # plot data
        ax_euler.plot(euler_list)
        ax_imp_euler.plot(imp_euler_list)
        ax_rk.plot(rk_list)
        ax_exact.plot(exact_list)

        # refresh canvas
        self.canvas_euler.draw()
        self.canvas_imp_euler.draw()
        self.canvas_rk.draw()
        self.canvas_exact.draw()

# diff. eq. y' = y/x - xe^(y/x)
def y_prime(x, y):
    return y / x - x * (pow(np.e, (y / x)))


def exact_solution(x):
    return (-x) * np.log(x)


exact_list = []  # exact solution
euler_list = []  # euler method
imp_euler_list = []  # improved euler method
rk_list = []  # runge-kutta

le_euler = []  # local error of euler method
le_imp_euler = []  # local error of improved euler method
le_rk = []  # local error of runge-kutta method


def compute_lists(X0, Y0, X, step):
    list_exact_solution(X0, X, step)
    list_approximated_solution(X0, Y0, X, step)


def list_exact_solution(X0, N, H):
    global exact_list
    exact_list = []
    for i in np.arange(X0, N + H, H):
        exact_list.append(exact_solution(i))


def list_approximated_solution(X0, Y0, N, H):
    # Euler method y_n = y_n-1 + h*f(n-1, y_n-1)
    # Improved Euler method y_n = y_n-1 + h/2 * (f(n-1, y_n-1) + f(n, y_n-1 + h*f(n-1, y_n-1)))
    # Runge-Kutta method y_n =

    global euler_list
    global imp_euler_list
    global rk_list

    global le_euler
    global le_imp_euler
    global le_rk

    euler_list = []
    imp_euler_list = []
    rk_list = []

    le_euler = []
    le_imp_euler = []
    le_rk = []

    euler_list.append(Y0)
    imp_euler_list.append(Y0)
    rk_list.append(Y0)

    le_euler.append(0)
    le_imp_euler.append(0)
    le_rk.append(0)

    for i in np.arange(X0 + H, N + H, H):
        e = euler_list[-1] + H * y_prime(i - H, euler_list[-1])
        i_e = imp_euler_list[-1] + 0.5 * H * (y_prime(i - H, imp_euler_list[-1])
                                              + y_prime(i - H, imp_euler_list[-1]
                                                        + h * y_prime(i - H, imp_euler_list[-1])))

        k1 = y_prime(i - H, rk_list[-1])
        k2 = y_prime(i - H / 2, rk_list[-1] + (H * k1) / 2)
        k3 = y_prime(i - H / 2, rk_list[-1] + (H * k2) / 2)
        k4 = y_prime(i, rk_list[-1] + H * k3)
        rk = rk_list[-1] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        exact = exact_solution(i)

        euler_list.append(e)
        imp_euler_list.append(i_e)
        rk_list.append(rk)

        le_euler.append(abs(exact - e))
        le_imp_euler.append(abs(exact - i_e))
        le_rk.append(abs(exact - rk))


# Blue-le_euler, Orange-le_imp_euler, Green-le_rk
# plt.figure(figsize=[10, 5])
# plt.plot(ax, le_euler, ax, le_imp_euler, ax, le_rk)
# plt.show()
#
# df = pd.DataFrame(zip(ax, euler_list, imp_euler_list, rk_list, exact_list,
#                       [h for i in np.arange(x0, X + h, h)], le_euler, le_imp_euler, le_rk),
#                   columns=['x', 'euler', 'imp_euler', 'runge-kutta', 'exact', 'h',
#                            'le euler', 'le imp euler', 'le runge-kutta'])


if __name__ == '__main__':
    y0 = 0  # intial value for y
    x0 = 1  # intial value for x
    X = 8
    h = 5  # step
    compute_lists(x0, y0, X, h)

    app = QApplication(sys.argv)

    main = Window()
    main.showMaximized()

    sys.exit(app.exec_())
