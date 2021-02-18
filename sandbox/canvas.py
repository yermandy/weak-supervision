import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")
from PyQt5 import QtWidgets, QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from milp import milp
from milp import objective
from sandbox import Point

class Canvas(FigureCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, X, K, parent=None, width=6, height=6):

        self.fig = Figure(figsize=(width, height))
        self.axes = self.fig.add_subplot(111)

        self.axes.grid(False)
        self.fig.tight_layout()

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.points = []
        self.colors = {1: 'r', 2: 'g', 3: 'b'}
        self.K = K

        self.show()

        bag_to_index = {}
        for i, k in enumerate(K):
            if k not in bag_to_index:
                bag_to_index[k] = []
            bag_to_index[k].append(i)

        self.bag_to_index = bag_to_index
        
        self.median = Point(self, 0.5, 0.5, 'black', draggable=False)
        self.milp_median = Point(self, 0.5, 0.5, 'magenta', draggable=False)
        self.plot_draggable_points(X, K)
    

    def update_median(self):
        X = []
        for p in self.points:
            x, y = p.point.center
            X.append([x, y])
        X = np.array(X)

        median = np.median(X, axis=0, keepdims=True).T
        milp_median, alphas = milp(X, self.K, False, return_alphas=True)
        milp_median = np.atleast_2d(milp_median).T

        for i, alpha in enumerate(alphas):
            if alpha == 1:
                self.points[i].point.set_alpha(0.75)
            else:
                self.points[i].point.set_alpha(0.25)


        print(f'{objective(self.bag_to_index, X, median):.4f} : {objective(self.bag_to_index, X, milp_median):.4f}')
        
        self.median.point.center = median.flatten()
        self.milp_median.point.center = milp_median.flatten()


    def plot_draggable_points(self, X, K):
        """Plot and define the 2 draggable points of the baseline"""
        
        for (x, y), k in zip(X, K):
            point = Point(self, x, y, self.colors[k])
            self.points.append(point)

        self.update_median()
        self.update_figure()


    def clear_figure(self):
        """Clear the graph"""

        self.axes.clear()
        self.axes.grid(False)
        del(self.points[:])
        self.update_figure()


    def update_figure(self):
        """Update the graph. Necessary, to call after each plot"""

        self.draw()


if __name__ == '__main__':

    X = []
    for i in range(6):
        X.append(np.random.rand(2))
    K = [1,1,2,2,3,3]

    app = QtWidgets.QApplication(sys.argv)
    Canvas(X, K)
    sys.exit(app.exec_())
