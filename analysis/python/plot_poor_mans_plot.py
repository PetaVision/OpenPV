
"""
Plot the highest activity of four different bar positionings
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import PVReadSparse as rs
import PVReadWeights as rw
import PVConversions as conv
import scipy.cluster.vq as sp
import math


"""
Show how to modify the coordinate formatter to report the image "z"
value of the nearest pixel given x and y
"""
extended = False
vmax = 100.0 # Hz

if len(sys.argv) < 22:
   print "usage: plot_avg_activity filename 1-32, [end_time step_time begin_time], test filename, On-weigh filename, Off-weight filename"
   sys.exit()

#if len(sys.argv) >= 6:
#   vmax = float(sys.argv[5])


a1 = rs.PVReadSparse(sys.argv[1], extended)
a2 = rs.PVReadSparse(sys.argv[2], extended)
a3 = rs.PVReadSparse(sys.argv[3], extended)
a4 = rs.PVReadSparse(sys.argv[4], extended)
a5 = rs.PVReadSparse(sys.argv[5], extended)
a6 = rs.PVReadSparse(sys.argv[6], extended)
a7 = rs.PVReadSparse(sys.argv[7], extended)
a8 = rs.PVReadSparse(sys.argv[8], extended)
a9 = rs.PVReadSparse(sys.argv[9], extended)
a10 = rs.PVReadSparse(sys.argv[10], extended)
a11 = rs.PVReadSparse(sys.argv[11], extended)
a12 = rs.PVReadSparse(sys.argv[12], extended)
a13 = rs.PVReadSparse(sys.argv[13], extended)
a14 = rs.PVReadSparse(sys.argv[14], extended)
a15 = rs.PVReadSparse(sys.argv[15], extended)
a16 = rs.PVReadSparse(sys.argv[16], extended)

end = int(sys.argv[17])
step = int(sys.argv[18])
begin = int(sys.argv[19])
endtest = end
steptest = step
begintest = begin
atest = rs.PVReadSparse(sys.argv[20], extended)
#zetest = rs.PVReadSparse(sys.argv[21], extended)
w =  rw.PVReadWeights(sys.argv[21])
wO = rw.PVReadWeights(sys.argv[22])


zerange = end
margin = 15

pa = []


print "(begin, end, step, max) == ", begin, end, step, vmax


for endtest in range(begintest+steptest, steptest+1, steptest):
   Atest = atest.avg_activity(begintest, endtest)
   lenofo = len(Atest)
   for i in range(lenofo):
      for j in range(lenofo):
         pa = np.append(pa, Atest[i,j])  
median = np.median(pa)
avg = np.mean(pa)

AW = np.zeros((lenofo, lenofo))
AWmin = np.zeros((lenofo, lenofo))

AWO = np.zeros((lenofo, lenofo))
SUMAW = np.zeros((lenofo, lenofo))

space = 1
nx  = w.nx
ny  = w.ny
nxp = w.nxp
nyp = w.nyp
nf = w.nf
d = np.zeros((5,5))
coord = 1

numpat = w.numPatches


res = 0
rmax = 0
rmin = 0
count = 0

#print "avg = ", avg
#print "median = ", median
#a2.rewind()
co = 0


for end in range(begin+step, step+1, step):
   A1 = a1.avg_activity(begin, end)
   A2 = a2.avg_activity(begin, end)
   A3 = a3.avg_activity(begin, end)
   A4 = a4.avg_activity(begin, end)
   A5 = a5.avg_activity(begin, end)
   A6 = a6.avg_activity(begin, end)
   A7 = a7.avg_activity(begin, end)
   A8 = a8.avg_activity(begin, end)
   A9 = a9.avg_activity(begin, end)
   A10 = a10.avg_activity(begin, end)
   A11 = a11.avg_activity(begin, end)
   A12 = a12.avg_activity(begin, end)
   A13 = a13.avg_activity(begin, end)
   A14 = a14.avg_activity(begin, end)
   A15 = a15.avg_activity(begin, end)
   A16 = a16.avg_activity(begin, end)


   AF = np.zeros((lenofo, lenofo))
   countpos = 0

   lenofo = len(A1)
   lenofb = lenofo * lenofo

   vmax = 0
   hmax = 0
   vdif = 0
   hdif = 0
   vcount = 0
   hcount = 0
   vd = 0
   hd = 0
   count = 0

   whis = np.zeros((16))

   plot = np.zeros((lenofo, lenofo))

   for i in range(lenofo):
      for j in range(lenofo):
         #print A1[i, j]
         check = [A1[i,j], A2[i,j], A3[i,j], A4[i,j], A5[i,j], A6[i,j], A7[i,j], A8[i,j], A9[i,j], A10[i,j], A11[i,j], A12[i,j], A13[i,j], A14[i,j], A15[i,j], A16[i,j]]
         narmax = np.argmax(check)  

         if narmax == 0:
            plot[i, j] = 0.3
         if narmax == 1:
            plot[i, j] = 0.3
         if narmax == 2:
            plot[i, j] = 0.3
         if narmax == 3:
            plot[i, j] = 0.3
         if narmax == 4:
            plot[i, j] = 0.3
         if narmax == 5:
            plot[i, j] = 0.3
         if narmax == 6:
            plot[i, j] = 0.3
         if narmax == 7:
            plot[i, j] = 0.3
         if narmax == 8:
            plot[i, j] = 0.7
         if narmax == 9:
            plot[i, j] = 0.7
         if narmax == 10:
            plot[i, j] = 0.7
         if narmax == 11:
            plot[i, j] = 0.7
         if narmax == 12:
            plot[i, j] = 0.7
         if narmax == 13:
            plot[i, j] = 0.7
         if narmax == 14:
            plot[i, j] = 0.7
         if narmax == 15:
            plot[i, j] = 0.7



   fig = plt.figure()
   ax = fig.add_subplot(111)
   ax.set_title('Feature Plot')
   ax.imshow(plot, cmap=cm.jet, interpolation='nearest', vmin=0., vmax=1)

   plt.show()
