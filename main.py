import sys
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QGridLayout, QHBoxLayout, QFormLayout, \
    QLineEdit, QLabel
from graph import Graph
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
        self.figure_error = plt.figure()

        # set the layout
        main_layout = QHBoxLayout()
        config_layout = QFormLayout()
        layout = QVBoxLayout()
        glayout = QGridLayout()

        b_add_euler = QPushButton("Euler")
        b_add_imp_euler = QPushButton("Improved Euler")
        b_add_rk = QPushButton("Runge-Kutta")
        b_add_exact = QPushButton("Exact")

        b_add_euler.setStyleSheet("background-color: #DCDDDF")
        b_add_imp_euler.setStyleSheet("background-color: #DCDDDF")
        b_add_rk.setStyleSheet("background-color: #DCDDDF")
        b_add_exact.setStyleSheet("background-color: #DCDDDF")

        b_add_euler.clicked.connect(self.euler_switch)
        b_add_imp_euler.clicked.connect(self.imp_euler_switch)
        b_add_rk.clicked.connect(self.rk_switch)
        b_add_exact.clicked.connect(self.exact_switch)

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas_function = FigureCanvas(self.figure_function)
        self.canvas_error = FigureCanvas(self.figure_error)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar_function = NavigationToolbar(self.canvas_function, self)
        self.toolbar_error = NavigationToolbar(self.canvas_error, self)

        self.toolbar_function.setStyleSheet("background-color: white")
        self.toolbar_error.setStyleSheet("background-color: white")

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

        glayout.addWidget(self.toolbar_error, 0, 1)
        glayout.addWidget(self.canvas_error, 1, 1)

        self.x0 = QLineEdit()
        self.y0 = QLineEdit()
        self.X = QLineEdit()
        self.h = QLineEdit()

        self.x0.setStyleSheet("background-color: #DCDDDF")
        self.y0.setStyleSheet("background-color: #DCDDDF")
        self.X.setStyleSheet("background-color: #DCDDDF")
        self.h.setStyleSheet("background-color: #DCDDDF")

        self.x0.setText("1")
        self.y0.setText("0")
        self.X.setText("8")
        self.h.setText("4")

        x0_text = QLabel('x0:')
        y0_text = QLabel('y0:')
        X_text = QLabel('X:')
        h_text = QLabel('h:')

        x0_text.setStyleSheet("color: #DCDDDF")
        y0_text.setStyleSheet("color: #DCDDDF")
        X_text.setStyleSheet("color: #DCDDDF")
        h_text.setStyleSheet("color: #DCDDDF")

        config_layout.addRow(x0_text, self.x0)
        config_layout.addRow(y0_text, self.y0)
        config_layout.addRow(X_text, self.X)
        config_layout.addRow(h_text, self.h)

        config_layout.addWidget(self.b_set)
        config_layout.addWidget(self.b_plot)

        config_layout.addWidget(b_add_euler)
        config_layout.addWidget(b_add_imp_euler)
        config_layout.addWidget(b_add_rk)
        config_layout.addWidget(b_add_exact)

        main_layout.addLayout(layout)
        main_layout.addLayout(config_layout)

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
        if self.x0.text() != "" and self.y0.text() != "" and self.X.text() != "" and self.h.text() != "":

            if float(self.x0.text()) == 0:
                self.x0.setText("Incorrect")
            elif float(self.X.text()) <= 2:
                self.X.setText("Incorrect")
            else:
                self.graph.set_params(float(self.x0.text()), float(self.y0.text()), float(self.X.text()),
                                      float(self.h.text()))
                self.plot()

    def plot(self):
        # clear all figures
        self.figure_function.clear()
        self.figure_error.clear()

        # create an axis
        ax_function = self.figure_function.add_subplot(111, xlabel="values of x", ylabel="values of y")
        ax_error = self.figure_error.add_subplot(111, xlabel="values of x", ylabel="values of y")

        # set titles
        ax_function.set(title='Graph of Exact and Approximation Functions')
        ax_error.set(title='Graph of Errors')

        # Plot chosen graphs
        if self.is_euler:
            func, error = self.graph.calc_euler()
            ax_function.plot(self.graph.axis, func, color="#F6D349", label='Euler')
            ax_error.plot(self.graph.axis, error, color="#F6D349", label='Euler')

        if self.is_imp_euler:
            func, error = self.graph.calc_imp_euler()
            ax_function.plot(self.graph.axis, func, color="#8FDDD3", label='Improved Euler')
            ax_error.plot(self.graph.axis, error, color="#8FDDD3", label='Improved Euler')

        if self.is_rk:
            func, error = self.graph.calc_rk()
            ax_function.plot(self.graph.axis, func, color="magenta", label='Runge-Kutta')
            ax_error.plot(self.graph.axis, error, color="magenta", label='Runge-Kutta')

        if self.is_exact:
            ax_function.plot(self.graph.e_axis, self.graph.calc_exact(), '--', color="red", label='Exact')

        # Display Legend of each graph
        ax_function.legend()
        ax_error.legend()

        # refresh canvas
        self.canvas_function.draw()
        self.canvas_error.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    graph = Window()
    graph.setStyleSheet("background-color: #1D222F")
    graph.showMaximized()
    sys.exit(app.exec_())
