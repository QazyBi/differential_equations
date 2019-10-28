import sys
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QGridLayout, QHBoxLayout, QFormLayout, \
    QLineEdit, QLabel
from graph import Graph
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class Window(QDialog):
    def __init__(self, is_graph=True, parent=None):
        super(Window, self).__init__(parent)

        # One window will be for displaying errors and one for approximation graphs
        self.is_graph = is_graph

        # Model, from MVC design pattern
        self.graph = Graph()

        if self.is_graph:
            # a figure instance to plot on
            self.figure_euler = plt.figure()
            self.figure_imp_euler = plt.figure()
            self.figure_rk = plt.figure()
            self.figure_exact = plt.figure()
        else:
            self.figure_euler_error = plt.figure()
            self.figure_imp_euler_error = plt.figure()
            self.figure_rk_error = plt.figure()

        # set the layout
        main_layout = QHBoxLayout()
        config_layout = QFormLayout()
        layout = QVBoxLayout()
        glayout = QGridLayout()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        if self.is_graph:
            self.canvas_euler = FigureCanvas(self.figure_euler)
            self.canvas_imp_euler = FigureCanvas(self.figure_imp_euler)
            self.canvas_rk = FigureCanvas(self.figure_rk)
            self.canvas_exact = FigureCanvas(self.figure_exact)
        else:
            self.canvas_euler_error = FigureCanvas(self.figure_euler_error)
            self.canvas_imp_euler_error = FigureCanvas(self.figure_imp_euler_error)
            self.canvas_rk_error = FigureCanvas(self.figure_rk_error)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        if self.is_graph:
            self.toolbar_euler = NavigationToolbar(self.canvas_euler, self)
            self.toolbar_imp_euler = NavigationToolbar(self.canvas_imp_euler, self)
            self.toolbar_rk = NavigationToolbar(self.canvas_rk, self)
            self.toolbar_exact = NavigationToolbar(self.canvas_exact, self)
        else:
            self.toolbar_euler_error = NavigationToolbar(self.canvas_euler_error, self)
            self.toolbar_imp_euler_error = NavigationToolbar(self.canvas_imp_euler_error, self)
            self.toolbar_rk_error = NavigationToolbar(self.canvas_rk_error, self)

        # Just some button connected to `plot` method
        self.b_plot = QPushButton('Plot ')
        self.b_set = QPushButton("Set")

        # Add action to buttons
        self.b_plot.clicked.connect(self.plot)
        self.b_set.clicked.connect(self.change_params)

        # Add widgets to layouts
        layout.addLayout(glayout)
        layout.addWidget(self.b_plot)

        if is_graph:
            glayout.addWidget(self.toolbar_euler, 0, 0)
            glayout.addWidget(self.canvas_euler, 1, 0)

            glayout.addWidget(self.toolbar_imp_euler, 0, 1)
            glayout.addWidget(self.canvas_imp_euler, 1, 1)

            glayout.addWidget(self.toolbar_rk, 2, 0)
            glayout.addWidget(self.canvas_rk, 3, 0)

            glayout.addWidget(self.toolbar_exact, 2, 1)
            glayout.addWidget(self.canvas_exact, 3, 1)
        else:
            glayout.addWidget(self.toolbar_euler_error, 0, 0)
            glayout.addWidget(self.canvas_euler_error, 1, 0)

            glayout.addWidget(self.toolbar_imp_euler_error, 0, 1)
            glayout.addWidget(self.canvas_imp_euler_error, 1, 1)

            glayout.addWidget(self.toolbar_rk_error, 2, 0)
            glayout.addWidget(self.canvas_rk_error, 3, 0)

        self.x0 = QLineEdit()
        self.y0 = QLineEdit()
        self.X = QLineEdit()
        self.h = QLineEdit()

        config_layout.addRow(QLabel('x0:'), self.x0)
        config_layout.addRow(QLabel('y0:'), self.y0)
        config_layout.addRow(QLabel('X:'), self.X)
        config_layout.addRow(QLabel('h:'), self.h)
        config_layout.addWidget(self.b_set)

        main_layout.addLayout(layout)
        main_layout.addLayout(config_layout)

        self.setLayout(main_layout)

    def change_params(self):
        if self.x0.text() != "" and self.y0.text() != "" and self.X.text() != "" and self.h.text() != "" and \
                self.x0.text().isdigit() and self.y0.text().isdigit() and self.X.text().isdigit() \
                and self.h.text().isdigit():
            self.graph.set_params(int(self.x0.text()), int(self.y0.text()), int(self.X.text()), int(self.h.text()))
            self.plot()

    def plot(self):
        # Calculate
        self.graph.calculate()

        if self.is_graph:
            # clear all figures
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
            ax_euler.plot(self.graph.euler_list, color="blue")
            ax_imp_euler.plot(self.graph.imp_euler_list, color="magenta")
            ax_rk.plot(self.graph.rk_list, color="red")
            ax_exact.plot(self.graph.exact_list, color="black")

            # refresh canvas
            self.canvas_euler.draw()
            self.canvas_imp_euler.draw()
            self.canvas_rk.draw()
            self.canvas_exact.draw()
        else:
            self.figure_euler_error.clear()
            self.figure_imp_euler_error.clear()
            self.figure_rk_error.clear()

            ax_euler_error = self.figure_euler_error.add_subplot(111)
            ax_imp_euler_error = self.figure_imp_euler_error.add_subplot(111)
            ax_rk_error = self.figure_rk_error.add_subplot(111)

            # plot data
            ax_euler_error.plot(self.graph.le_imp_euler, color="blue")
            ax_imp_euler_error.plot(self.graph.le_imp_euler, color="magenta")
            ax_rk_error.plot(self.graph.le_rk, color="red")

            # refresh canvas
            self.canvas_euler_error.draw()
            self.canvas_imp_euler_error.draw()
            self.canvas_rk_error.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    graph = Window(True)
    error = Window(False)

    graph.showMaximized()
    error.showMaximized()

    sys.exit(app.exec_())
