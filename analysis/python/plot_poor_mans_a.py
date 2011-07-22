
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

if len(sys.argv) < 38:
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
a17 = rs.PVReadSparse(sys.argv[17], extended)
a18 = rs.PVReadSparse(sys.argv[18], extended)
a19 = rs.PVReadSparse(sys.argv[19], extended)
a20 = rs.PVReadSparse(sys.argv[20], extended)
a21 = rs.PVReadSparse(sys.argv[21], extended)
a22 = rs.PVReadSparse(sys.argv[22], extended)
a23 = rs.PVReadSparse(sys.argv[23], extended)
a24 = rs.PVReadSparse(sys.argv[24], extended)
a25 = rs.PVReadSparse(sys.argv[25], extended)
a26 = rs.PVReadSparse(sys.argv[26], extended)
a27 = rs.PVReadSparse(sys.argv[27], extended)
a28 = rs.PVReadSparse(sys.argv[28], extended)
a29 = rs.PVReadSparse(sys.argv[29], extended)
a30 = rs.PVReadSparse(sys.argv[30], extended)
a31 = rs.PVReadSparse(sys.argv[31], extended)
a32 = rs.PVReadSparse(sys.argv[32], extended)




end = int(sys.argv[33])
step = int(sys.argv[34])
begin = int(sys.argv[35])
endtest = end
steptest = step
begintest = begin
atest = rs.PVReadSparse(sys.argv[36], extended)
#zetest = rs.PVReadSparse(sys.argv[21], extended)
w =  rw.PVReadWeights(sys.argv[37])
wO = rw.PVReadWeights(sys.argv[38])


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

print "avg = ", avg
print "median = ", median
#a2.rewind()
co = 0
for g in range(2):
   if g == 0:
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
         A17 = a17.avg_activity(begin, end)
         A18 = a18.avg_activity(begin, end)
         A19 = a19.avg_activity(begin, end)
         A20 = a20.avg_activity(begin, end)
         A21 = a21.avg_activity(begin, end)
         A22 = a22.avg_activity(begin, end)
         A23 = a23.avg_activity(begin, end)
         A24 = a24.avg_activity(begin, end)
         A25 = a25.avg_activity(begin, end)
         A26 = a26.avg_activity(begin, end)
         A27 = a27.avg_activity(begin, end)
         A28 = a28.avg_activity(begin, end)
         A29 = a29.avg_activity(begin, end)
         A30 = a30.avg_activity(begin, end)
         A31 = a31.avg_activity(begin, end)
         A32 = a32.avg_activity(begin, end)


         AF = np.zeros((lenofo, lenofo))
         countpos = 0

         lenofo = len(A1)
         lenofb = lenofo * lenofo
         beingplotted = []

         for i in range(lenofo):
            for j in range(lenofo):
               #print A1[i, j]
               check = [A1[i,j], A2[i,j], A3[i,j], A4[i,j], A5[i,j], A6[i,j], A7[i,j], A8[i,j], A9[i,j], A10[i,j], A11[i,j], A12[i,j], A13[i,j], A14[i,j], A15[i,j], A16[i,j], A17[i,j], A18[i,j], A19[i,j], A20[i,j], A21[i,j], A22[i,j], A23[i,j], A24[i,j], A25[i,j], A26[i,j], A27[i,j], A28[i,j], A29[i,j], A30[i,j], A31[i,j], A32[i,j]] 

               checkmax = np.max(check)
               checkmin = np.min(check)
 
               rmax += checkmax
               rmin += checkmin
               #print check
               #print "checkmax = ", checkmax

               re = (checkmax - checkmin)/(checkmax + checkmin)

               if checkmax == 0 and checkmin == 0:
                  print "both equal 0"
                  re = 0
               res += re
               count += 1



     


         fmax = rmax / count
         fmin = rmin / count
         final = res / count
         print
         print
         print
         print
         print "result = ", final
         print "average max = ", fmax
         print "average min = ", fmin
