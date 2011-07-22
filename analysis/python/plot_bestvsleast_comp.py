
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
   print "usage: plot_avg_activity filename1, filename2, filename3, filename4, filename5, filename6, filename7, filename8, filename9, filename10, filename11, filename12, filename13, filename14, filename15, filename16 [end_time step_time begin_time], test filename, On-weigh filename, Off-weight filename"
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
roc1 = np.loadtxt(sys.argv[23], delimiter=';')
roclen1 = np.shape(roc1)[0] 
roclen2 = np.shape(roc1)[1]


#sys.exit()
roc2 = np.loadtxt(sys.argv[24], delimiter=';')
roc3 = np.loadtxt(sys.argv[25], delimiter=';')
roc4 = np.loadtxt(sys.argv[26], delimiter=';')
roc5 = np.loadtxt(sys.argv[27], delimiter=';')
roc6 = np.loadtxt(sys.argv[28], delimiter=';')
roc7 = np.loadtxt(sys.argv[29], delimiter=';')
roc8 = np.loadtxt(sys.argv[30], delimiter=';')
roc9 = np.loadtxt(sys.argv[31], delimiter=';')
roc10 = np.loadtxt(sys.argv[32], delimiter=';')
roc11 = np.loadtxt(sys.argv[33], delimiter=';')
roc12 = np.loadtxt(sys.argv[34], delimiter=';')
roc13 = np.loadtxt(sys.argv[35], delimiter=';')
roc14 = np.loadtxt(sys.argv[36], delimiter=';')
roc15 = np.loadtxt(sys.argv[37], delimiter=';')
roc16 = np.loadtxt(sys.argv[38], delimiter=';')


zerange = end
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
count6 = 0
count7 = 0
count8 = 0
count9 = 0
count10 = 0
count11 = 0
count12 = 0
count13 = 0
count14 = 0
count15 = 0
count16 = 0
count17 = 0
count18 = 0
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
d = np.zeros((4,4))
coord = 1
nx_im = nx * (nxp + space) + space
ny_im = ny * (nyp + space) + space
numpat = w.numPatches

im = np.zeros((nx_im, ny_im))
im[:,:] = (w.max - w.min) / 2.
countnum = 0

im2 = np.zeros((nx_im, ny_im))
im2[:,:] = (w.max - w.min) / 2.

thecount=0
A1pos = np.array([0,0])
minpos = np.array([0,0])
minlist = np.array([0,0])
countpos = 0

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
         AF = np.zeros((lenofo, lenofo))
         countpos = 0

         lenofo = len(A1)
         lenofb = lenofo * lenofo
         beingplotted = []
         for i in range(lenofo):
            for j in range(lenofo):
               #print A1[i, j]
               check = [A1[i,j], A2[i,j], A3[i,j], A4[i,j], A5[i,j], A6[i,j], A7[i,j], A8[i,j], A9[i,j], A10[i,j], A11[i,j], A12[i,j], A13[i,j], A14[i,j], A15[i,j], A16[i,j]] 

               checkmax = np.max(check)
               checkmin = np.min(check)
               wheremax = np.argmax(check)
               wheremin = np.argmin(check)
               #print "wheremin = ", wheremin

               half = checkmax / 2.0
               sort = np.sort(check)
               co = 0
               if wheremin == 0:
                  AWmin[i, j] = 1
               if wheremin == 1:
                  AWmin[i, j] = 2
               if wheremin == 2:
                  AWmin[i, j] = 3
               if wheremin == 3:
                  AWmin[i, j] = 4
               if wheremin == 4:
                  AWmin[i, j] = 5
               if wheremin == 5:
                  AWmin[i, j] = 6
               if wheremin == 6:
                  AWmin[i, j] = 7
               if wheremin == 7:
                  AWmin[i, j] = 8
               if wheremin == 8:
                  AWmin[i, j] = 9
               if wheremin == 9:
                  AWmin[i, j] = 10
               if wheremin == 10:
                  AWmin[i, j] = 11
               if wheremin == 11:
                  AWmin[i, j] = 12
               if wheremin == 12:
                  AWmin[i, j] = 13
               if wheremin == 13:
                  AWmin[i, j] = 14
               if wheremin == 14:
                  AWmin[i, j] = 15
               if wheremin == 15:
                  AWmin[i, j] = 16



               if wheremax == 0:
                  AW[i, j] = 1
               if wheremax == 1:
                  AW[i, j] = 2
               if wheremax == 2:
                  AW[i, j] = 3
               if wheremax == 3:
                  AW[i, j] = 4
               if wheremax == 4:
                  AW[i, j] = 5
               if wheremax == 5:
                  AW[i, j] = 6
               if wheremax == 6:
                  AW[i, j] = 7
               if wheremax == 7:
                  AW[i, j] = 8
               if wheremax == 8:
                  AW[i, j] = 9
               if wheremax == 9:
                  AW[i, j] = 10
               if wheremax == 10:
                  AW[i, j] = 11
               if wheremax == 11:
                  AW[i, j] = 12
               if wheremax == 12:
                  AW[i, j] = 13
               if wheremax == 13:
                  AW[i, j] = 14
               if wheremax == 14:
                  AW[i, j] = 15
               if wheremax == 15:
                  AW[i, j] = 16
               

               #print AF[i, j]
               #print "check = ", sort
               #print "half = ", half
               countnum += 1
               if i > (margin-1) and i < (w.nx - margin):
                  if j > (margin-1) and j < (w.ny - margin):
                           #print "QUICKLY"
                     if countpos == 0:
                        A1pos = [i, j]
                        minpos = [i, j]
                        A1list = [wheremax]
                        minlist = [wheremin]
                     else:
                        A1pos = np.vstack((A1pos, [i, j]))
                        minpos = np.vstack((minpos, [i,j]))
                        A1list = np.vstack((A1list, wheremax))
                        minlist = np.vstack((minlist, wheremin))
                     countpos+=1

      print "shape of roc1 = ", np.shape(roc1)

#############################################################
      for i in range(roclen1):
         print "i = ", i
         A1t = np.zeros((roclen2))
         mint = np.zeros((roclen2))
         #print "A1t = ", A1t
         for j in range(roclen2):
            #print "j = ", j
            if A1list[j] == 0:
               A1t[j]+=roc1[i,j]
            if A1list[j] == 1:
               A1t[j]+=roc2[i,j]
            if A1list[j] == 2:
               A1t[j]+=roc3[i,j]
            if A1list[j] == 3:
               A1t[j]+=roc4[i,j]
            if A1list[j] == 4:
               A1t[j]+=roc5[i,j]
            if A1list[j] == 5:
               A1t[j]+=roc6[i,j]
            if A1list[j] == 6:
               A1t[j]+=roc7[i,j]
            if A1list[j] == 7:
               A1t[j]+=roc8[i,j]
            if A1list[j] == 8:
               A1t[j]+=roc9[i,j]
            if A1list[j] == 9:
               A1t[j]+=roc10[i,j]
            if A1list[j] == 10:
               #print "roc11[i,j] = ", roc11[i, j]
               #print "a1t[j] = ", A1t[j]

               A1t[j]+=roc11[i,j]
            if A1list[j] == 11:
               A1t[j]+=roc12[i,j]
            if A1list[j] == 12:
               A1t[j]+=roc13[i,j]
            if A1list[j] == 13:
               A1t[j]+=roc14[i,j]
            if A1list[j] == 14:
               A1t[j]+=roc15[i,j]
            if A1list[j] == 15:
               A1t[j]+=roc16[i,j]
      
            if minlist[j] == 0:
               mint[j]+=roc1[i,j]
            if minlist[j] == 1:
               mint[j]+=roc2[i,j]
            if minlist[j] == 2:
               mint[j]+=roc3[i,j]
            if minlist[j] == 3:
               mint[j]+=roc4[i,j]
            if minlist[j] == 4:
               mint[j]+=roc5[i,j]
            if minlist[j] == 5:
               mint[j]+=roc6[i,j]
            if minlist[j] == 6:
               mint[j]+=roc7[i,j]
            if minlist[j] == 7:
               mint[j]+=roc8[i,j]
            if minlist[j] == 8:
               mint[j]+=roc9[i,j]
            if minlist[j] == 9:
               mint[j]+=roc10[i,j]
            if minlist[j] == 10:
               mint[j]+=roc11[i,j]
            if minlist[j] == 11:
               mint[j]+=roc12[i,j]
            if minlist[j] == 12:
               mint[j]+=roc13[i,j]
            if minlist[j] == 13:
               mint[j]+=roc14[i,j]
            if minlist[j] == 14:
               mint[j]+=roc15[i,j]
            if minlist[j] == 15:
               mint[j]+=roc16[i,j]
         if i == 0:
            A1q = A1t
            minq = mint
         else:
            A1q = np.vstack((A1q, A1t))
            minq = np.vstack((minq, mint))


      print "a1q = ", A1q
      print "minq = ", minq

      print "a1q shape = ", np.shape(A1q)
      print "minq shape = ", np.shape(minq)

      mave = np.average(A1q, axis = 0)
      iave = np.average(minq, axis = 0)
      ans = []
      for i in range(len(mave)):
         res = (mave[i]-iave[i]) / (mave[i] + iave[i])
         ans = np.append(ans, res)

      np.savetxt("b-vs-w.txt", A1qf, fmt='%d', delimiter = ';')        

      sys.exit()
