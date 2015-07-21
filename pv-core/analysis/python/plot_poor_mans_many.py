
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

if len(sys.argv) < 20:
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
#zetest = rs.PVReadSparse(sys.argv[21], extended)

lenofo = a1.nx


zerange = end
margin = 15

pa = []


print "(begin, end, step, max) == ", begin, end, step, vmax

space = 1

d = np.zeros((5,5))
coord = 1


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

   vcount = 0
   hcount = 0
   vlcount = 0
   hlcount = 0

   his = np.zeros((8))
   checkv1 = np.zeros((16))
   checkv2 = np.zeros((16))
   checkv3 = np.zeros((16))
   checkv4 = np.zeros((16))
   checkv5 = np.zeros((16))
   checkv6 = np.zeros((16))
   checkv7 = np.zeros((16))
   checkv8 = np.zeros((16))
   checkh1 = np.zeros((16))
   checkh2 = np.zeros((16))
   checkh3 = np.zeros((16))
   checkh4 = np.zeros((16))
   checkh5 = np.zeros((16))
   checkh6 = np.zeros((16))
   checkh7 = np.zeros((16))
   checkh8 = np.zeros((16))

   countv1 = 0
   countv2 = 0
   countv3 = 0
   countv4 = 0
   countv5 = 0
   countv6 = 0
   countv7 = 0
   countv8 = 0
   counth1 = 0
   counth2 = 0
   counth3 = 0
   counth4 = 0
   counth5 = 0
   counth6 = 0
   counth7 = 0
   counth8 = 0


   for i in range(lenofo):
      for j in range(lenofo):
         if 20 < i < 108 and 20 < j < 108:
            #print A1[i, j]
            check = [A1[i,j], A2[i,j], A3[i,j], A4[i,j], A5[i,j], A6[i,j], A7[i,j], A8[i,j], A9[i,j], A10[i,j], A11[i,j], A12[i,j], A13[i,j], A14[i,j], A15[i,j], A16[i,j]]

            count += 1
            checkmax = np.max(check)
            checkmin = np.min(check)

            if checkmax == 0 and checkmin == 0:
               print "both equal 0"


          
            am = np.argmax(check)
            if am == 0:
               checkv1 = np.add(checkv1, check)
               countv1 += 1
            if am == 1:
               checkv2 = np.add(checkv2, check)
               countv2 += 1
            if am == 2:
               checkv3 = np.add(checkv3, check)
               countv3 += 1
            if am == 3:
               checkv4 = np.add(checkv4, check)
               countv4 += 1
            if am == 4:
               checkv5 = np.add(checkv5, check)
               countv5 += 1
            if am == 5:
               checkv6 = np.add(checkv6, check)
               countv6 += 1
            if am == 6:
               checkv7 = np.add(checkv7, check)
               countv7 += 1
            if am == 7:
               checkv8 = np.add(checkv8, check)
               countv8 += 1
####
            if am == 8:
               checkh1 = np.add(checkh1, check)
               counth1 += 1
            if am == 9:
               checkh2 = np.add(checkh2, check)
               counth2 += 1
            if am == 10:
               checkh3 = np.add(checkh3, check)
               counth3 += 1
            if am == 11:
               checkh4 = np.add(checkh4, check)
               counth4 += 1
            if am == 12:
               checkh5 = np.add(checkh5, check)
               counth5 += 1
            if am == 13:
               checkh6 = np.add(checkh6, check)
               counth6 += 1
            if am == 14:
               checkh7 = np.add(checkh7, check)
               counth7 += 1
            if am == 15:
               checkh8 = np.add(checkh8, check)
               counth8 += 1



   checkv1 = checkv1 / countv1
   checkv2 = checkv2 / countv2
   checkv3 = checkv3 / countv3
   checkv4 = checkv4 / countv4
   checkv5 = checkv5 / countv5
   checkv6 = checkv6 / countv6
   checkv7 = checkv7 / countv7
   checkv8 = checkv8 / countv8
   checkh1 = checkh1 / counth1
   checkh2 = checkh2 / counth2
   checkh3 = checkh3 / counth3
   checkh4 = checkh4 / counth4
   checkh5 = checkh5 / counth5
   checkh6 = checkh6 / counth6
   checkh7 = checkh7 / counth7
   checkh8 = checkh8 / counth8

             
   loc = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5 ]

   fig = plt.figure()

   ax = fig.add_subplot(211)
   width = 1
   ax.bar(loc, checkv1, width=1.0, bottom=0, color=cm.jet(1.0))
   ax.bar(loc, checkv2, width=0.875, bottom=0, color=cm.jet(0.875))
   ax.bar(loc, checkv3, width=0.75, bottom=0, color=cm.jet(0.75))
   ax.bar(loc, checkv4, width=0.625, bottom=0, color=cm.jet(0.625))
   ax.bar(loc, checkv5, width=0.5, bottom=0, color=cm.jet(0.5))
   ax.bar(loc, checkv6, width=0.375, bottom=0, color=cm.jet(0.375))
   ax.bar(loc, checkv7, width=0.25, bottom=0, color=cm.jet(0.25))
   ax.bar(loc, checkv8, width=0.125, bottom=0, color=cm.jet(0.125))


   ax.set_title("Vertical")

   ax2 = fig.add_subplot(212)
   ax2.bar(loc, checkh1, width=1.0, bottom=0, color=cm.jet(1.0))
   ax2.bar(loc, checkh2, width=0.875, bottom=0, color=cm.jet(0.875))
   ax2.bar(loc, checkh3, width=0.75, bottom=0, color=cm.jet(0.75))
   ax2.bar(loc, checkh4, width=0.625, bottom=0, color=cm.jet(0.625))
   ax2.bar(loc, checkh5, width=0.5, bottom=0, color=cm.jet(0.5))
   ax2.bar(loc, checkh6, width=0.375, bottom=0, color=cm.jet(0.375))
   ax2.bar(loc, checkh7, width=0.25, bottom=0, color=cm.jet(0.25))
   ax2.bar(loc, checkh8, width=0.125, bottom=0, color=cm.jet(0.125))



   ax2.set_xlabel("vcount1 = %i vcount2 = %i vcount3 = %i vcount4 = %i vcount5 = %i vcount6 = %i vcount7 = %i vcount8 = %i \n hcount1 = %i hcount2 = %i hcount3 = %i hcount4 = %i hcount5 = %i hcount6 = %i hcount7 = %i hcount8 = %i " %(countv1,countv2,countv3,countv4,countv5,countv6,countv7,countv8,counth1,counth2,counth3,counth4,counth5,counth6,counth7,counth8))
   ax2.set_title("Horizontal")


   plt.show()
