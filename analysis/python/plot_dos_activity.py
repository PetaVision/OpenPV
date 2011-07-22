"""
Compare Horizontal and Vertical Activity Files
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
        z = A1[row,col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f'%(x, y)


extended = False
begin = 0
end = 10000
step = 1000
vmax = 100.0 # Hz

if len(sys.argv) < 6:
   print "usage: plot_avg_activity Horizontal-filename Vertical-filename [end_time step_time begin_time] test_filename"
   sys.exit()

#if len(sys.argv) >= 6:
#   vmax = float(sys.argv[5])

print "(begin, end, step, max) == ", begin, end, step, vmax

a1 = rs.PVReadSparse(sys.argv[1], extended)
a2 = rs.PVReadSparse(sys.argv[2], extended)
end = int(sys.argv[3])
step = int(sys.argv[4])
begin = int(sys.argv[5])
endtest = end
steptest = step
begintest = begin
atest = rs.PVReadSparse(sys.argv[6], extended)
count1 = 0
count2 = 0
count3 = 0
pa = []

for endtest in range(begintest+steptest, endtest, steptest):
   Atest = atest.avg_activity(begintest, endtest)
   lenofo = len(Atest)
   for i in range(lenofo):
      for j in range(lenofo):
         pa = np.append(pa, Atest[i,j])
median = np.median(pa)
avg = np.mean(pa)
print "avg = ", avg
print "median = ", median

for end in range(begin+step, end, step):
   A1 = a1.avg_activity(begin, end)
   A2 = a2.avg_activity(begin, end)
   AF = A1
   lenofo = len(A1)
   for i in range(lenofo):
      for j in range(lenofo):
         #print "A1", A1[i, j]
         #print "A2", A2[i, j]
         check = [A1[i,j], A2[i,j]]
         checkmax = np.max(check)
         doublecheck = 0
         if A1[i,j] > median or A2[i,j] > median:
            if A2[i, j] > A1[i, j]:
               count2 += 1
               AF[i, j] = 1.0
               doublecheck += 1
            else:
               AF[i, j] = 0.0
               count1 += 1
               doublecheck += 1

            if doublecheck > 1:
               AF[i,j] = 0.6
               count3 += 1
         else:
            AF[i,j] = 0.48
         #print count
   print "Horizontal = ", count1
   print "Vertical = ", count2
   print "both 0 = ", count3
   numrows, numcols = A1.shape

   min = np.min(A1)
   max = np.max(A1)

   s = np.zeros(numcols)
   for col in range(numcols):
       s[col] = np.sum(A1[:,col])
   s = s/numrows

   fig = plt.figure()
   ax = fig.add_subplot(2,1,1)

   ax.set_xlabel('BLUE=HORIZONTAL RED=VERTICAL GREEN=Both below median\n Horizontal=%1.0i, Vertical=%1.0i, Both=%1.0i' %(count1, count2, count3))
   ax.set_ylabel('Ky GLOBAL')
   ax.set_title('Activity: min=%1.1f, max=%1.1f time=%d' %(min, max, a1.time))
   ax.format_coord = format_coord
   ax.imshow(AF, cmap=cm.jet, interpolation='nearest', vmin=0., vmax=1)


   ax = fig.add_subplot(2,1,2)
   ax.set_ylabel('Ky Avg Activity')
   ax.plot(s, 'o')
   ax.set_ylim(0.0, vmax)

   #attempt at colorbar
   #cax = fig.add_axes([0.85, 0.1, 0.075, 0.8]) 
   #fig.colorbar(A, cax=cax)

   plt.show()

#end fig loop
