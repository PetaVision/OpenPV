
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

   vhis = np.zeros((8))
   hhis = np.zeros((8))
   vlhis = np.zeros((8))
   hlhis = np.zeros((8))



   for i in range(lenofo):
      for j in range(lenofo):
         if 20 < i < 108 and 20 < j < 108:
            #print A1[i, j]
            checkv = [A1[i,j], A2[i,j], A3[i,j], A4[i,j], A5[i,j], A6[i,j], A7[i,j], A8[i,j]] 
            checkh = [A9[i,j], A10[i,j], A11[i,j], A12[i,j], A13[i,j], A14[i,j], A15[i,j], A16[i,j]]

            count += 1
            checkmaxv = np.max(checkv)
            checkminv = np.min(checkv)
 
            checkmaxh = np.max(checkh)
            checkminh = np.min(checkh)

            if checkmaxv == 0 and checkminv == 0:
               print "both equal 0"
            elif checkmaxh == 0 and checkminh == 0:
               print "both equal 0"

          
            if checkmaxv > checkmaxh:
               am = np.argmax(checkv)
               if am == 0:
                  checkv = [A6[i,j], A7[i,j], A8[i,j], A1[i,j], A2[i,j], A3[i,j], A4[i,j], A5[i,j]] 
               if am == 1:
                  checkv = [A7[i,j], A8[i,j], A1[i,j], A2[i,j], A3[i,j], A4[i,j], A5[i,j], A6[i,j]] 
               if am == 2:
                  checkv = [A8[i,j], A1[i,j], A2[i,j], A3[i,j], A4[i,j], A5[i,j], A6[i,j], A7[i,j]] 
               if am == 3:
                  checkv = [A1[i,j], A2[i,j], A3[i,j], A4[i,j], A5[i,j], A6[i,j], A7[i,j], A8[i,j]] 
               if am == 4:
                  checkv = [A2[i,j], A3[i,j], A4[i,j], A5[i,j], A6[i,j], A7[i,j], A8[i,j], A1[i,j]] 
               if am == 5:
                  checkv = [A3[i,j], A4[i,j], A5[i,j], A6[i,j], A7[i,j], A8[i,j], A1[i,j], A2[i,j]] 
               if am == 6:
                  checkv = [A4[i,j], A5[i,j], A6[i,j], A7[i,j], A8[i,j], A1[i,j], A2[i,j], A3[i,j]] 
               if am == 7:
                  checkv = [A5[i,j], A6[i,j], A7[i,j], A8[i,j], A1[i,j], A2[i,j], A3[i,j], A4[i,j]] 
               vhis = np.add(vhis, checkv)
               vcount += 1

               hlm = np.argmax(checkh)
               if hlm == 0:
                  checkh = [A14[i,j], A15[i,j], A16[i,j], A9[i,j], A10[i,j], A11[i,j], A12[i,j], A13[i,j]] 
               if hlm == 1:
                  checkh = [A15[i,j], A16[i,j], A9[i,j], A10[i,j], A11[i,j], A12[i,j], A13[i,j], A14[i,j]] 
               if hlm == 2:
                  checkh = [A16[i,j], A9[i,j], A10[i,j], A11[i,j], A12[i,j], A13[i,j], A14[i,j], A15[i,j]] 
               if hlm == 3:
                  checkh = [A9[i,j], A10[i,j], A11[i,j], A12[i,j], A13[i,j], A14[i,j], A15[i,j], A16[i,j]] 
               if hlm == 4:
                  checkh = [A10[i,j], A11[i,j], A12[i,j], A13[i,j], A14[i,j], A15[i,j], A16[i,j], A9[i,j]] 
               if hlm == 5:
                  checkh = [A11[i,j], A12[i,j], A13[i,j], A14[i,j], A15[i,j], A16[i,j], A9[i,j], A10[i,j]] 
               if hlm == 6:
                  checkh = [A12[i,j], A13[i,j], A14[i,j], A15[i,j], A16[i,j], A9[i,j], A10[i,j], A11[i,j]] 
               if hlm == 7:
                  checkh = [A13[i,j], A14[i,j], A15[i,j], A16[i,j], A9[i,j], A10[i,j], A11[i,j], A12[i,j]] 
               hlhis = np.add(hlhis, checkh)
               hlcount += 1



            if checkmaxv < checkmaxh:

               hm = np.argmax(checkh)
               if hm == 0:
                  checkh = [A14[i,j], A15[i,j], A16[i,j], A9[i,j], A10[i,j], A11[i,j], A12[i,j], A13[i,j]] 
               if hm == 1:
                  checkh = [A15[i,j], A16[i,j], A9[i,j], A10[i,j], A11[i,j], A12[i,j], A13[i,j], A14[i,j]] 
               if hm == 2:
                  checkh = [A16[i,j], A9[i,j], A10[i,j], A11[i,j], A12[i,j], A13[i,j], A14[i,j], A15[i,j]] 
               if hm == 3:
                  checkh = [A9[i,j], A10[i,j], A11[i,j], A12[i,j], A13[i,j], A14[i,j], A15[i,j], A16[i,j]] 
               if hm == 4:
                  checkh = [A10[i,j], A11[i,j], A12[i,j], A13[i,j], A14[i,j], A15[i,j], A16[i,j], A9[i,j]] 
               if hm == 5:
                  checkh = [A11[i,j], A12[i,j], A13[i,j], A14[i,j], A15[i,j], A16[i,j], A9[i,j], A10[i,j]] 
               if hm == 6:
                  checkh = [A12[i,j], A13[i,j], A14[i,j], A15[i,j], A16[i,j], A9[i,j], A10[i,j], A11[i,j]] 
               if hm == 7:
                  checkh = [A13[i,j], A14[i,j], A15[i,j], A16[i,j], A9[i,j], A10[i,j], A11[i,j], A12[i,j]] 
               hhis = np.add(hhis, checkh)
               hcount += 1

               alm = np.argmax(checkv)
               if alm == 0:
                  checkv = [A6[i,j], A7[i,j], A8[i,j], A1[i,j], A2[i,j], A3[i,j], A4[i,j], A5[i,j]] 
               if alm == 1:
                  checkv = [A7[i,j], A8[i,j], A1[i,j], A2[i,j], A3[i,j], A4[i,j], A5[i,j], A6[i,j]] 
               if alm == 2:
                  checkv = [A8[i,j], A1[i,j], A2[i,j], A3[i,j], A4[i,j], A5[i,j], A6[i,j], A7[i,j]] 
               if alm == 3:
                  checkv = [A1[i,j], A2[i,j], A3[i,j], A4[i,j], A5[i,j], A6[i,j], A7[i,j], A8[i,j]] 
               if alm == 4:
                  checkv = [A2[i,j], A3[i,j], A4[i,j], A5[i,j], A6[i,j], A7[i,j], A8[i,j], A1[i,j]] 
               if alm == 5:
                  checkv = [A3[i,j], A4[i,j], A5[i,j], A6[i,j], A7[i,j], A8[i,j], A1[i,j], A2[i,j]] 
               if alm == 6:
                  checkv = [A4[i,j], A5[i,j], A6[i,j], A7[i,j], A8[i,j], A1[i,j], A2[i,j], A3[i,j]] 
               if alm == 7:
                  checkv = [A5[i,j], A6[i,j], A7[i,j], A8[i,j], A1[i,j], A2[i,j], A3[i,j], A4[i,j]] 
               vlhis = np.add(vlhis, checkv)
               vlcount += 1



   vhis = vhis / vcount
   hhis = hhis / hcount
   vlhis = vlhis / vlcount
   hlhis = hlhis / hlcount

             
   loc = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]

   fig = plt.figure()

   ax = fig.add_subplot(211)
   width = 1
   ax.bar(loc, vhis, width=width, bottom=0, color='b')
   ax.bar(loc, hlhis, width=0.75, bottom=0, color='r')

   ax.set_title("Vertical")

   ax2 = fig.add_subplot(212)
   ax2.bar(loc, hhis, width=width, bottom=0, color='b')
   ax2.bar(loc, vlhis, width=0.75, bottom=0, color='r')

   ax2.set_xlabel("vcount = %i  hcount = %i" %(vcount, hcount))
   ax2.set_title("Horizontal")


   plt.show()
