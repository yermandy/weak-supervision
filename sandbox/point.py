import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D


class Point:

    # http://stackoverflow.com/questions/21654008/matplotlib-drag-overlapping-points-interactively

    lock = None #  only one can be animated at a time

    def __init__(self, parent, x=0.1, y=0.1, color='r', size=0.02, draggable=True):

        self.parent = parent
        self.point = patches.Ellipse((x, y), size, size, fc=color, alpha=0.7, edgecolor=color)
        parent.fig.axes[0].add_patch(self.point)
        self.x = x
        self.y = y
        self.press = None
        self.background = None
        if draggable:
            self.connect()


    def connect(self):
        """Connect to all the events we need"""

        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)


    def on_press(self, event):

        if event.inaxes != self.point.axes: return
        if Point.lock is not None: return
        contains, attrd = self.point.contains(event)
        if not contains: return
        self.press = (self.point.center), event.xdata, event.ydata
        Point.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.point.figure.canvas
        axes = self.point.axes
        self.point.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.point.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.point)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)


    def on_motion(self, event):

        if Point.lock is not self:
            return
        if event.inaxes != self.point.axes: return
        self.point.center, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.point.center = (self.point.center[0]+dx, self.point.center[1]+dy)
        self.x = self.point.center[0]
        self.y = self.point.center[1]

        canvas = self.point.figure.canvas
        axes = self.point.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.point)

        # blit just the redrawn area
        canvas.blit(axes.bbox)


    def on_release(self, event):
        """On release we reset the press data"""

        if Point.lock is not self:
            return

        self.press = None
        Point.lock = None

        # turn off the rect animation property and reset the background
        self.point.set_animated(False)

        self.background = None

        # recalculate median
        self.parent.update_median()

        # redraw the full figure
        self.point.figure.canvas.draw()



    def disconnect(self):
        """Disconnect all the stored connection ids"""

        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)
