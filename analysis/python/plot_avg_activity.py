"""
Make a histogram of normally distributed random numbers and plot the
analytic PDF over it
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import PVReadSparse as rs

def format_coord(x, y):
    col = int(x+0.5)
    row = int(y+0.5)
    if col>=0 and col<numcols and row>=0 and row<numrows:
        z = A[row,col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f'%(x, y)

"""
Show how to modify the coordinate formatter to report the image "z"
value of the nearest pixel given x and y
"""
extended = False
begin = 0.0
end = 500.0

if len(sys.argv) < 3:
   print "usage: plot_avg_activity filename extended_flag [end_time]"
   sys.exit()

if len(sys.argv) >= 3:
   extended = bool(sys.argv[2])

if len(sys.argv) >= 4:
   end = float(sys.argv[3])

activ = rs.PVReadSparse(sys.argv[1], extended)

for end in range(5000, 5000, 5000):
   A = activ.avg_activity(begin, end)

   numrows, numcols = A.shape

   min = np.min(A)
   max = np.max(A)

   fig = plt.figure()
   ax = fig.add_subplot(111)

   ax.set_xlabel('Kx GLOBAL')
   ax.set_ylabel('Ky GLOBAL')
   ax.set_title( 'Activity: min=%1.1f, max=%1.1f time=%f' %(min, max, activ.time) )
   ax.format_coord = format_coord

   ax.imshow(A, cmap=cm.jet, interpolation='nearest', vmin=0., vmax=100.)
   plt.show()
#end fig loop

print 'finished'
