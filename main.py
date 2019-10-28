import sys
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QGridLayout, QHBoxLayout, QFormLayout, QLineEdit, QLabel
from graph import Graph
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class Window(QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # Model, from MVC design pattern
        self.graph = Graph()

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
        main_layout = QHBoxLayout()
        config_layout = QFormLayout()
        layout = QVBoxLayout()
        glayout = QGridLayout()

        layout.addWidget(self.toolbar)
        layout.addLayout(glayout)
        layout.addWidget(self.b_plot)

        glayout.addWidget(self.canvas_euler, 0, 0)
        glayout.addWidget(self.canvas_imp_euler, 0, 1)
        glayout.addWidget(self.canvas_rk, 1, 0)
        glayout.addWidget(self.canvas_exact, 1, 1)

        config_layout.addRow(QLabel('x0:'), QLineEdit())
        config_layout.addRow(QLabel('y0:'), QLineEdit())
        config_layout.addRow(QLabel('X:'), QLineEdit())
        config_layout.addRow(QLabel('h:'), QLineEdit())

        set_button = QPushButton("Set")
        config_layout.addWidget(set_button)
        main_layout.addLayout(layout)
        main_layout.addLayout(config_layout)

        self.setLayout(main_layout)

    def plot(self):

        # Calculate
        self.graph.calculate()
        self.graph.calc_exact()

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


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.showMaximized()

    sys.exit(app.exec_())


