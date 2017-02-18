import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import os

class SplineBuilder:
    def __init__(self, line, fig):
        self.line = line
        self.fig = fig
        self.xs = None
        self.ys = None
        self.tck = None
        self.cidclick = fig.canvas.mpl_connect('button_press_event', self)
        self.cidkey = fig.canvas.mpl_connect('key_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.inaxes != self.line.axes:
            # click off canvas to end spline
            print '---------------------------------------------------------------------------'
            save_track = raw_input("type \'y\' to save the current track. Otherwise type \'q\' to quit or \'d\' to keep drawing: ")
            if save_track == 'y':
                filename = raw_input("Enter the name of the track: ")
                path = '/Users/kyle/Documents/SCHOOL/Stanford/Research/RoboRace/workspace/trackSplines/'+filename
                if os.path.exists(path):
                    over_write = raw_input("a track with that filename has already been saved. Type \'y\' to overwrite it. Otherwise type \'q\': ")
                    if over_write != 'y':
                        return
                else:
                     os.mkdir(path)
                np.save(path +'/x', np.array(self.xs))
                np.save(path +'/y', np.array(self.ys))
                np.save(path +'/tck', np.array(self.tck))
                print filename, "saved! exiting..."
                exit(0)
            if save_track == 'n':
                print "exiting without saving..."
                exit(0)
            return

        if self.xs == None:
            self.xs = list([event.xdata])
            self.ys = list([event.ydata])
        else:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)

        if len(self.xs) < 4:
            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()
        else:
            # generates spline points by interpolation
            self.tck, u = interpolate.splprep([self.xs, self.ys], s=0) # s is smoothing factor
            unew = np.arange(0, 1.01, 0.005)
            # evaluate spline at locations specified by unew
            out = interpolate.splev(unew, self.tck)

            self.line.set_data(out[0], out[1])
            self.line.figure.canvas.draw()


fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111)

ax.set_title('Click to build spline')
line, = ax.plot([0], [0])  # empty line
splinebuilder = SplineBuilder(line,fig)

# define axis range 
plt.axis('equal')
plt.xticks(np.arange(0, 10, 0.5))
plt.yticks(np.arange(0, 10, 0.5))
plt.ylim(-1,10)
plt.xlim(-1,10)
ax.grid(zorder=0, color='gray',which='both')

plt.show()