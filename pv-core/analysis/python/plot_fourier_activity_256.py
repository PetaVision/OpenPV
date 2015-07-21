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
import radialProfile
import pylab as py

def format_coord(x, y):
   col = int(x+0.5)
   row = int(y+0.5)
   if coord == 3:
      check = ((x - 0.5) % 16)
      if check < 4:
         x2 = ((x - 0.5) % 16) - 7 + (x / 16.0)
         y2 = ((y - 0.5) % 16) - 7 + (y / 16.0) 
      elif check < 10:
         x2 = ((x - 0.5) % 16) - 7.5 + (x / 16.0)
         y2 = ((y - 0.5) % 16) - 7.5 + (y / 16.0) 
      else:
         x2 = ((x - 0.5) % 16) - 8 + (x / 16.0)
         y2 = ((y - 0.5) % 16) - 8 + (y / 16.0) 
      x = (x / 16.0)
      y = (y / 16.0)
      

      if col>=0 and col<numcols and row>=0 and row<numrows:
         z = P[row,col]
         return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
      else:
         return 'x=%1.4d, y=%1.4d, x2=%1.4d, y2=%1.4d'%(int(x), int(y), int(x2), int(y2))      

   if coord == 1:
      x2 = (x / 20.0)
      y2 = (y / 20.0)
      x = (x / 5.0)
      y = (y / 5.0)
      if col>=0 and col<numcols and row>=0 and row<numrows:
         z = P[row,col]
         return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
      else:
         return 'x=%1.4d, y=%1.4d, x2=%1.4d, y2=%1.4d'%(int(x), int(y), int(x2), int(y2))

"""
Show how to modify the coordinate formatter to report the image "z"
value of the nearest pixel given x and y
"""
extended = False
vmax = 100.0 # Hz

if len(sys.argv) < 26:
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
a17 = rs.PVReadSparse(sys.argv[17], extended)
a18 = rs.PVReadSparse(sys.argv[18], extended)
a19 = rs.PVReadSparse(sys.argv[19], extended)
a20 = rs.PVReadSparse(sys.argv[20], extended)


end = int(sys.argv[21])
step = int(sys.argv[22])
begin = int(sys.argv[23])
endtest = end
steptest = step
begintest = begin
atest = rs.PVReadSparse(sys.argv[24], extended)
w = rw.PVReadWeights(sys.argv[25])
wO = rw.PVReadWeights(sys.argv[26])
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
margin = 0
histo = np.zeros((1, 20))



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
AWO = np.zeros((lenofo, lenofo))
SUMAW = np.zeros((lenofo, lenofo))
countpos = 0
space = 1
nx  = w.nx
ny  = w.ny
nxp = w.nxp
nyp = w.nyp
nf = w.nf
d = np.zeros((5,5))
coord = 1
nx_im = nx * (nxp + space) + space
ny_im = ny * (nyp + space) + space
numpat = w.numPatches

print "avg = ", avg
print "median = ", median
#a2.rewind()
co = 0
for g in range(2):
   if g == 0:
      for end in range(begin+step, step+1, step):
         countpos = 0
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

         AF = np.zeros((lenofo, lenofo))

         vertmax = []      
         horimax = []

         vertmax = np.max(A1)
         vertmax = np.append(vertmax, np.max(A2))
         vertmax = np.append(vertmax, np.max(A3))
         vertmax = np.append(vertmax, np.max(A4))
         vertmax = np.append(vertmax, np.max(A5))
         vertmax = np.append(vertmax, np.max(A6))
         vertmax = np.append(vertmax, np.max(A7))
         vertmax = np.append(vertmax, np.max(A8))
         vertmax = np.append(vertmax, np.max(A9))
         vertmax = np.append(vertmax, np.max(A10))

         horimax = np.max(A11)
         horimax = np.append(horimax, np.max(A12))
         horimax = np.append(horimax, np.max(A13))
         horimax = np.append(horimax, np.max(A14))
         horimax = np.append(horimax, np.max(A15))
         horimax = np.append(horimax, np.max(A16))
         horimax = np.append(horimax, np.max(A17))
         horimax = np.append(horimax, np.max(A18))
         horimax = np.append(horimax, np.max(A19))
         horimax = np.append(horimax, np.max(A20))

         vmax = np.max(vertmax)
         hmax = np.max(horimax)
         

         lenofo = len(A1)
         lenofb = lenofo * lenofo
         beingplotted = []
         for i in range(lenofo):
            for j in range(lenofo):
               #print A1[i, j]
               check = [A1[i,j], A2[i,j], A3[i,j], A4[i,j], A5[i,j], A6[i,j], A7[i,j], A8[i,j], A9[i,j], A10[i,j], A11[i,j], A12[i,j], A13[i,j], A14[i,j], A15[i,j], A16[i,j], A17[i,j], A18[i,j], A19[i,j], A20[i,j] ] 
               vcheck = [A1[i,j], A2[i,j], A3[i,j], A4[i,j], A5[i,j], A6[i,j], A7[i,j], A8[i,j] , A9[i,j] , A10[i,j]] 
               hcheck = [A11[i,j], A12[i,j], A13[i,j], A14[i,j], A15[i,j], A16[i,j], A17[i,j], A18[i,j] , A19[i,j] , A20[i,j]] 


               checkmax = np.max(check)
               wheremax = np.argmax(check)
               half = checkmax / 2.0
               sort = np.sort(check)
               co = 0
               if wheremax <= 7:
                  AW[i, j] = 128 + ((128 * np.max(vcheck)) / float(vmax))
               if wheremax > 7:
                  AW[i, j] = 128 - ((128 * np.max(hcheck)) / float(hmax))


print AW
print "vmax = ", vmax
print "hmax = ", hmax



F1 = np.fft.fft2(AW)
F2 = np.fft.fftshift(F1)
psd2D = np.abs(F2)**2
psd1D = radialProfile.azimuthalAverage(psd2D)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(np.log10(AW), cmap=py.cm.Greys)



fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.imshow(np.log10(psd2D))

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.semilogy(psd1D)
ax3.set_xlabel('Spatial Frequency')
ax3.set_ylabel('Power Spectrum')

plt.show()



sys.exit()
