import sys
from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, QGridLayout, QLineEdit, QLabel

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT  # as NavigationToolbar

from MyFunction import MyFunction
from EulerMethod import EulerApproximation
from ImprovedEulerMethod import ImpEulerApproximation
from RKMethod import RKApproximation


class NavigationToolbar(NavigationToolbar2QT):
    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar2QT.toolitems if
                 t[0] in ('Home', 'Pan', 'Zoom')]


class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        self.is_exact = True

        # Create models
        self.graph = MyFunction()
        self.graph_euler = EulerApproximation(self.graph)
        self.graph_imp_euler = ImpEulerApproximation(self.graph)
        self.graph_rk = RKApproximation(self.graph)

        # a figure instance to plot on
        self.figure_function = plt.figure(figsize=(6, 3))
        self.figure_local_error = plt.figure(figsize=(6, 3))
        self.figure_global_error = plt.figure(figsize=(6, 3))

        # Create layouts
        self.config_layout = QGridLayout()
        self.grid_layout = QGridLayout()

        # Create buttons
        self.b_add_euler = QPushButton("Euler")
        self.b_add_imp_euler = QPushButton("Improved Euler")
        self.b_add_rk = QPushButton("Runge-Kutta")
        self.b_add_exact = QPushButton("Exact")
        self.b_plot = QPushButton('Plot')
        self.b_set = QPushButton("Set")

        # Create canvases
        self.canvas_function = FigureCanvas(self.figure_function)
        self.canvas_local_error = FigureCanvas(self.figure_local_error)
        self.canvas_global_error = FigureCanvas(self.figure_global_error)

        self.toolbar_function = NavigationToolbar(self.canvas_function, self)
        self.toolbar_local_error = NavigationToolbar(self.canvas_local_error, self)
        self.toolbar_global_error = NavigationToolbar(self.canvas_global_error, self)
        # Create text fields
        self.x0 = QLineEdit()
        self.y0 = QLineEdit()
        self.X = QLineEdit()
        self.n = QLineEdit()
        self.n_start = QLineEdit()
        self.n_end = QLineEdit()

        # Create label for text fields
        self.x0_text = QLabel('x0')
        self.y0_text = QLabel('y0:')
        self.X_text = QLabel('X:')
        self.n_text = QLabel('N:')
        self.n_start_text = QLabel('N start:')
        self.n_end_text = QLabel('N end:')

        # Create list of objects
        self.models = [self.graph_euler, self.graph_imp_euler, self.graph_rk]
        self.all_models = [self.graph_euler, self.graph_imp_euler, self.graph_rk]
        self.canvases = [self.canvas_function, self.canvas_local_error, self.canvas_global_error]

        self.label_list = [self.x0_text, self.y0_text, self.X_text, self.n_text, self.n_start_text, self.n_end_text]
        self.line_edit_list = [self.x0, self.y0, self.X, self.n, self.n_start, self.n_end]
        self.button_list = [self.b_plot, self.b_set, self.b_add_euler, self.b_add_imp_euler, self.b_add_rk,
                            self.b_add_exact]

        # call functions
        self.customize_config_layout()
        self.customize_grid_layout()
        self.customize_ui()
        self.add_ui_functionality()
        self.setLayout(self.grid_layout)

    def customize_grid_layout(self):
        self.grid_layout.addWidget(self.canvas_function, 0, 0)
        self.grid_layout.addWidget(self.canvas_local_error, 0, 1)
        self.grid_layout.addWidget(self.canvas_global_error, 1, 0)
        self.grid_layout.addItem(self.config_layout)
        self.grid_layout.setSpacing(1)

    def customize_config_layout(self):
        for i in range(len(self.label_list)):
            self.config_layout.addWidget(self.label_list[i], i, 0)
            self.config_layout.addWidget(self.line_edit_list[i], i, 1)
            self.config_layout.addWidget(self.button_list[i], i, 2)
        self.config_layout.addWidget(self.toolbar_function)
        self.config_layout.addWidget(self.toolbar_local_error)
        self.config_layout.addWidget(self.toolbar_global_error)

    def customize_ui(self):
        for button in self.button_list:
            button.setStyleSheet("background-color: #DCDDDF")
            button.setFixedSize(150, 30)

        for line in self.line_edit_list:
            line.setFixedSize(50, 30)
            line.setStyleSheet("background-color: #DCDDDF")

        for label in self.label_list:
            label.setFixedSize(50, 30)
            label.setStyleSheet("color: #DCDDDF")

        self.x0.setText("1")
        self.y0.setText("0")
        self.X.setText("8")
        self.n.setText("4")
        self.n_start.setText("1")
        self.n_end.setText("5")

    def add_ui_functionality(self):
        # Set action to buttons
        self.b_add_euler.clicked.connect(self.euler_switch)
        self.b_add_imp_euler.clicked.connect(self.imp_euler_switch)
        self.b_add_rk.clicked.connect(self.rk_switch)
        self.b_add_exact.clicked.connect(self.exact_switch)
        self.b_plot.clicked.connect(self.plot)
        self.b_set.clicked.connect(self.change_params)

    def euler_switch(self):
        if self.models.__contains__(self.graph_euler):
            self.models.remove(self.graph_euler)
        else:
            self.models.append(self.graph_euler)
        self.plot()

    def imp_euler_switch(self):
        if self.models.__contains__(self.graph_imp_euler):
            self.models.remove(self.graph_imp_euler)
        else:
            self.models.append(self.graph_imp_euler)
        self.plot()

    def rk_switch(self):
        if self.models.__contains__(self.graph_rk):
            self.models.remove(self.graph_rk)
        else:
            self.models.append(self.graph_rk)
        self.plot()

    def exact_switch(self):
        self.is_exact = not self.is_exact
        self.plot()

    def change_params(self):
        if self.x0.text() != "" and self.y0.text() != "" and self.X.text() != "" and self.n.text() != "" \
                and self.n_start.text() != "" and self.n_end.text() != "":

            if float(self.x0.text()) == 0:
                self.x0.setText("Incorrect")
            elif abs(float(self.X.text()) - abs(float(self.x0.text()))) < 2:
                self.X.setText("Incorrect")
            else:
                for model in self.all_models:
                    model.graph.set_params(float(self.x0.text()), float(self.y0.text()), float(self.X.text()),
                                           int(self.n.text()))
                    model.set_params(None, None, None, n_start=int(self.n_start.text()), n_end=int(self.n_end.text()))

                self.plot()

    def plot(self):
        # clear all figures
        self.figure_function.clear()
        self.figure_local_error.clear()
        self.figure_global_error.clear()

        # create an axis
        ax_function = self.figure_function.add_subplot(xlabel="values of x", ylabel="values of y",
                                                       title="Graph of Exact and Approximation Functions")

        ax_local_error = self.figure_local_error.add_subplot(xlabel="values of x", ylabel="values of local error",
                                                             title='Graph of Local Errors')

        ax_global_error = self.figure_global_error.add_subplot(xlabel="values of N", ylabel="values of global error",
                                                               title='Graph of Global Errors')

        for model in self.models:
            func = model.calculate_approximation(model.graph.h)
            local_error = model.calculate_local_error()
            global_error = model.calculate_global_error()
            axis = model.axis(model.graph.h)
            axis_global_error = model.axis_global_error()

            ax_function.plot(axis, func, label=model.label, color=model.color)
            ax_local_error.plot(axis, local_error, label=model.label, color=model.color)
            ax_global_error.plot(axis_global_error, global_error, label=model.label, color=model.color)

        if self.is_exact:
            ax_function.plot(self.graph_euler.graph.e_axis, self.graph.calculate_exact(), '--', color="red",
                             label='Exact')

        # Display Legend of each graph
        ax_function.legend()
        ax_local_error.legend()
        ax_global_error.legend()
        plt.tight_layout()

        # refresh canvas
        for canvas in self.canvases:
            canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    graph = Window()
    graph.setStyleSheet("background-color: #1D222F")
    graph.showMaximized()
    sys.exit(app.exec_())
