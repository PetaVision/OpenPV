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
begin = 0
end = 10000
step = 1000
vmax = 100.0 # Hz

if len(sys.argv) < 3:
   print "usage: plot_avg_activity filename [end_time step_time]"
   sys.exit()

if len(sys.argv) >= 3:
   end = int(sys.argv[2])

if len(sys.argv) >= 4:
   step = int(sys.argv[3])

if len(sys.argv) >= 5:
   vmax = float(sys.argv[4])

print "(begin, end, step, max) == ", begin, end, step, vmax

activ = rs.PVReadSparse(sys.argv[1], extended)

for end in range(begin+step, end, step):
   A = activ.avg_activity(begin, end)

   numrows, numcols = A.shape

   min = np.min(A)
   max = np.max(A)

   s = np.zeros(numcols)
   for col in range(numcols):
       s[col] = np.sum(A[:,col])
   s = s/numrows

   fig = plt.figure()
   ax = fig.add_subplot(2,1,1)

   ax.set_xlabel('Kx GLOBAL')
   ax.set_ylabel('Ky GLOBAL')
   ax.set_title('Activity: min=%1.1f, max=%1.1f time=%d' %(min, max, activ.time))
   ax.format_coord = format_coord
   ax.imshow(A, cmap=cm.jet, interpolation='nearest', vmin=0., vmax=vmax)

   ax = fig.add_subplot(2,1,2)
   ax.set_ylabel('Ky Avg Activity')
   ax.plot(s, 'o')
   ax.set_ylim(0.0, vmax)

   plt.show()

#end fig loop
