import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


# load file
print '===================================='
filename = raw_input('What track would you like to load? ')
path = '/Users/kyle/Documents/SCHOOL/Stanford/Research/RoboRace/workspace/trackSplines/'+filename+'/tck.npy'
print '===================================='
print 'loading ', filename, '...'

tck = np.load(path)

# plot spline
unew = np.arange(0, 1.01, 0.005)
out = interpolate.splev(unew, tck)

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111)
ax.set_title('click to build spline')
line, = ax.plot(out[0], out[1])  # empty line

# define axis range 
plt.axis('equal')
plt.ylim(-1,10)
plt.xlim(-1,10)

plt.show()

