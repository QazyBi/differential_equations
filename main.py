import sys
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QGridLayout, QHBoxLayout, QFormLayout, \
    QLineEdit, QLabel
from graph import Graph
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class Window(QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        self.is_euler = True
        self.is_imp_euler = True
        self.is_rk = True
        self.is_exact = True

        # Model, from MVC design pattern
        self.graph = Graph()

        # a figure instance to plot on
        self.figure_function = plt.figure()
        self.figure_local_error = plt.figure()
        self.figure_global_error = plt.figure()

        # set the layout
        main_layout = QHBoxLayout()
        config_layout = QGridLayout()
        layout = QVBoxLayout()
        glayout = QGridLayout()

        self.b_add_euler = QPushButton("Euler")
        self.b_add_imp_euler = QPushButton("Improved Euler")
        self.b_add_rk = QPushButton("Runge-Kutta")
        self.b_add_exact = QPushButton("Exact")

        self.b_add_euler.setStyleSheet("background-color: #DCDDDF")
        self.b_add_imp_euler.setStyleSheet("background-color: #DCDDDF")
        self.b_add_rk.setStyleSheet("background-color: #DCDDDF")
        self.b_add_exact.setStyleSheet("background-color: #DCDDDF")

        self.b_add_euler.clicked.connect(self.euler_switch)
        self.b_add_imp_euler.clicked.connect(self.imp_euler_switch)
        self.b_add_rk.clicked.connect(self.rk_switch)
        self.b_add_exact.clicked.connect(self.exact_switch)

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas_function = FigureCanvas(self.figure_function)
        self.canvas_local_error = FigureCanvas(self.figure_local_error)
        self.canvas_global_error = FigureCanvas(self.figure_global_error)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar_function = NavigationToolbar(self.canvas_function, self)
        self.toolbar_local_error = NavigationToolbar(self.canvas_local_error, self)
        self.toolbar_global_error = NavigationToolbar(self.canvas_global_error, self)

        self.toolbar_function.setStyleSheet("background-color: white")
        self.toolbar_local_error.setStyleSheet("background-color: white")
        self.toolbar_global_error.setStyleSheet("background-color: white")

        # Just some button connected to `plot` method
        self.b_plot = QPushButton('Plot')
        self.b_set = QPushButton("Set")

        self.b_plot.setStyleSheet("background-color: #DCDDDF")
        self.b_set.setStyleSheet("background-color: #DCDDDF")
        # Add action to buttons
        self.b_plot.clicked.connect(self.plot)
        self.b_set.clicked.connect(self.change_params)

        # Add widgets to layouts
        layout.addLayout(glayout)

        glayout.addWidget(self.toolbar_function, 0, 0)
        glayout.addWidget(self.canvas_function, 1, 0)

        glayout.addWidget(self.toolbar_local_error, 0, 1)
        glayout.addWidget(self.canvas_local_error, 1, 1)

        glayout.addWidget(self.toolbar_global_error, 2, 0)
        glayout.addWidget(self.canvas_global_error, 3, 0)

        glayout.addItem(config_layout)
        self.x0 = QLineEdit()
        self.y0 = QLineEdit()
        self.X = QLineEdit()
        self.h = QLineEdit()
        self.h_start = QLineEdit()
        self.h_end = QLineEdit()

        self.x0.setFixedSize(50, 30)
        self.y0.setFixedSize(50, 30)
        self.X.setFixedSize(50, 30)
        self.h.setFixedSize(50, 30)
        self.h_start.setFixedSize(50, 30)
        self.h_end.setFixedSize(50, 30)

        self.b_plot.setFixedSize(150, 20)
        self.b_set.setFixedSize(150, 20)
        self.b_add_exact.setFixedSize(150, 20)
        self.b_add_imp_euler.setFixedSize(150, 20)
        self.b_add_euler.setFixedSize(150, 20)
        self.b_add_imp_euler.setFixedSize(150, 20)
        self.b_add_rk.setFixedSize(150, 20)

        self.x0.setStyleSheet("background-color: #DCDDDF")
        self.y0.setStyleSheet("background-color: #DCDDDF")
        self.X.setStyleSheet("background-color: #DCDDDF")
        self.h.setStyleSheet("background-color: #DCDDDF")
        self.h_start.setStyleSheet("background-color: #DCDDDF")
        self.h_end.setStyleSheet("background-color: #DCDDDF")

        self.x0.setText("1")
        self.y0.setText("0")
        self.X.setText("8")
        self.h.setText("4")
        self.h_start.setText("1")
        self.h_end.setText("5")

        x0_text = QLabel('x0:')
        y0_text = QLabel('y0:')
        X_text = QLabel('X:')
        h_text = QLabel('N:')
        h_start_text = QLabel('h start:')
        h_end_text = QLabel('h end:')

        x0_text.setFixedSize(20, 30)
        y0_text.setFixedSize(20, 30)
        X_text.setFixedSize(20, 30)
        h_text.setFixedSize(20, 30)
        h_start_text.setFixedSize(55, 30)
        h_end_text.setFixedSize(50, 30)

        x0_text.setStyleSheet("color: #DCDDDF")
        y0_text.setStyleSheet("color: #DCDDDF")
        X_text.setStyleSheet("color: #DCDDDF")
        h_text.setStyleSheet("color: #DCDDDF")
        h_start_text.setStyleSheet('color: #DCDDDF')
        h_end_text.setStyleSheet('color: #DCDDDF')

        config_layout.addWidget(x0_text, 0, 0)
        config_layout.addWidget(self.x0, 0, 1)

        config_layout.addWidget(y0_text, 1, 0)
        config_layout.addWidget(self.y0, 1, 1)

        config_layout.addWidget(X_text, 2, 0)
        config_layout.addWidget(self.X, 2, 1)

        config_layout.addWidget(h_text, 3, 0)
        config_layout.addWidget(self.h, 3, 1)

        config_layout.addWidget(h_start_text, 4, 0)
        config_layout.addWidget(self.h_start, 4, 1)

        config_layout.addWidget(h_end_text, 5, 0)
        config_layout.addWidget(self.h_end, 5, 1)

        config_layout.addWidget(self.b_set, 0, 2)
        config_layout.addWidget(self.b_plot, 1, 2)

        config_layout.addWidget(self.b_add_euler, 2, 2)
        config_layout.addWidget(self.b_add_imp_euler, 3, 2)
        config_layout.addWidget(self.b_add_rk, 4, 2)
        config_layout.addWidget(self.b_add_exact, 5, 2)

        main_layout.addLayout(layout)
        # main_layout.addLayout(config_layout)

        self.setLayout(main_layout)

    def euler_switch(self):
        self.is_euler = not self.is_euler
        self.plot()

    def imp_euler_switch(self):
        self.is_imp_euler = not self.is_imp_euler
        self.plot()

    def rk_switch(self):
        self.is_rk = not self.is_rk
        self.plot()

    def exact_switch(self):
        self.is_exact = not self.is_exact
        self.plot()

    def change_params(self):
        if self.x0.text() != "" and self.y0.text() != "" and self.X.text() != "" and self.h.text() != "" \
                and self.h_start.text() != "" and self.h_end.text() != "":

            if float(self.x0.text()) == 0:
                self.x0.setText("Incorrect")
            elif float(self.X.text()) <= 2:
                self.X.setText("Incorrect")
            else:
                self.graph.set_params(float(self.x0.text()), float(self.y0.text()), float(self.X.text()),
                                      float(self.h.text()), float(self.h_start.text()), float(self.h_end.text()))
                self.plot()

    def plot(self):
        # clear all figures
        self.figure_function.clear()
        self.figure_local_error.clear()
        self.figure_global_error.clear()

        # create an axis
        ax_function = self.figure_function.add_subplot(111, xlabel="values of x", ylabel="values of y")
        ax_local_error = self.figure_local_error.add_subplot(111, xlabel="values of x", ylabel="values of y")
        ax_global_error = self.figure_global_error.add_subplot(111, xlabel="values of h", ylabel="values of y")

        # set titles
        ax_function.set(title='Graph of Exact and Approximation Functions')
        ax_local_error.set(title='Graph of Local Errors')
        ax_global_error.set(title='Graph of Global Errors')

        # Plot chosen graphs
        if self.is_euler:
            func, local_error = self.graph.calc_euler()
            ax_function.plot(self.graph.axis, func, color="#F6D349", label='Euler')
            ax_local_error.plot(self.graph.axis, local_error, color="#F6D349", label='Euler')
            ax_global_error.plot(self.graph.g_axis, self.graph.calc_global_euler(), color="#F6D349", label='Euler')

        if self.is_imp_euler:
            func, local_error = self.graph.calc_imp_euler()
            ax_function.plot(self.graph.axis, func, color="#8FDDD3", label='Improved Euler')
            ax_local_error.plot(self.graph.axis, local_error, color="#8FDDD3", label='Improved Euler')
            ax_global_error.plot(self.graph.g_axis, self.graph.calc_global_imp_euler(), color="#8FDDD3",
                                 label='Improved Euler')

        if self.is_rk:
            func, local_error = self.graph.calc_rk()
            ax_function.plot(self.graph.axis, func, color="magenta", label='Runge-Kutta')
            ax_local_error.plot(self.graph.axis, local_error, color="magenta", label='Runge-Kutta')
            ax_global_error.plot(self.graph.g_axis, self.graph.calc_global_rk(), color="magenta", label='Runge-Kutta')

        if self.is_exact:
            ax_function.plot(self.graph.e_axis, self.graph.calc_exact(), '--', color="red", label='Exact')

        # Display Legend of each graph
        ax_function.legend()
        ax_local_error.legend()
        ax_global_error.legend()

        # refresh canvas
        self.canvas_function.draw()
        self.canvas_local_error.draw()
        self.canvas_global_error.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    graph = Window()
    graph.setStyleSheet("background-color: #1D222F")
    graph.showMaximized()
    sys.exit(app.exec_())
