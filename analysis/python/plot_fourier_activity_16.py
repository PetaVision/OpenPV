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

         print "a1 = ", A1
         print
         print "a1 max = ", np.max(A1)
         A1 = A1 / np.max(A1)
         print "2nd a1 = ", A1

         A2 = A2 / np.max(A2)
         A3 = A3 / np.max(A3)
         A4 = A4 / np.max(A4)
         A5 = A5 / np.max(A5)
         A6 = A6 / np.max(A6)
         A7 = A7 / np.max(A7)
         A8 = A8 / np.max(A8)
         A9 = A9 / np.max(A9)
         A10 = A10 / np.max(A10)
         A11 = A11 / np.max(A11)
         A12 = A12 / np.max(A12)
         A13 = A13 / np.max(A13)
         A14 = A14 / np.max(A14)
         A15 = A15 / np.max(A15)
         A16 = A16 / np.max(A16)
         A17 = A17 / np.max(A17)
         A18 = A18 / np.max(A18)
         A19 = A19 / np.max(A19)
         A20 = A20 / np.max(A20)


print A1
sys.exit()

A1F1 = np.fft.fft2(A1)
A1F2 = np.fft.fftshift(A1F1)
A1psd2D = np.abs(A1F2)**2
A1psd1D = radialProfile.azimuthalAverage(A1psd2D)

A2F1 = np.fft.fft2(A2)
A2F2 = np.fft.fftshift(A2F1)
A2psd2D = np.abs(A2F2)**2
A2psd1D = radialProfile.azimuthalAverage(A2psd2D)

A3F1 = np.fft.fft2(A3)
A3F2 = np.fft.fftshift(A3F1)
A3psd2D = np.abs(A3F2)**2
A3psd1D = radialProfile.azimuthalAverage(A3psd2D)

A4F1 = np.fft.fft2(A4)
A4F2 = np.fft.fftshift(A4F1)
A4psd2D = np.abs(A4F2)**2
A4psd1D = radialProfile.azimuthalAverage(A4psd2D)

A5F1 = np.fft.fft2(A5)
A5F2 = np.fft.fftshift(A5F1)
A5psd2D = np.abs(A5F2)**2
A5psd1D = radialProfile.azimuthalAverage(A5psd2D)

A6F1 = np.fft.fft2(A6)
A6F2 = np.fft.fftshift(A6F1)
A6psd2D = np.abs(A6F2)**2
A6psd1D = radialProfile.azimuthalAverage(A6psd2D)

A7F1 = np.fft.fft2(A7)
A7F2 = np.fft.fftshift(A7F1)
A7psd2D = np.abs(A7F2)**2
A7psd1D = radialProfile.azimuthalAverage(A7psd2D)

A8F1 = np.fft.fft2(A8)
A8F2 = np.fft.fftshift(A8F1)
A8psd2D = np.abs(A8F2)**2
A8psd1D = radialProfile.azimuthalAverage(A8psd2D)

A9F1 = np.fft.fft2(A9)
A9F2 = np.fft.fftshift(A9F1)
A9psd2D = np.abs(A9F2)**2
A9psd1D = radialProfile.azimuthalAverage(A9psd2D)

A10F1 = np.fft.fft2(A10)
A10F2 = np.fft.fftshift(A10F1)
A10psd2D = np.abs(A10F2)**2
A10psd1D = radialProfile.azimuthalAverage(A10psd2D)

A11F1 = np.fft.fft2(A11)
A11F2 = np.fft.fftshift(A11F1)
A11psd2D = np.abs(A11F2)**2
A11psd1D = radialProfile.azimuthalAverage(A11psd2D)

A12F1 = np.fft.fft2(A12)
A12F2 = np.fft.fftshift(A12F1)
A12psd2D = np.abs(A12F2)**2
A12psd1D = radialProfile.azimuthalAverage(A12psd2D)

A13F1 = np.fft.fft2(A13)
A13F2 = np.fft.fftshift(A13F1)
A13psd2D = np.abs(A13F2)**2
A13psd1D = radialProfile.azimuthalAverage(A13psd2D)

A14F1 = np.fft.fft2(A14)
A14F2 = np.fft.fftshift(A14F1)
A14psd2D = np.abs(A14F2)**2
A14psd1D = radialProfile.azimuthalAverage(A14psd2D)

A15F1 = np.fft.fft2(A15)
A15F2 = np.fft.fftshift(A15F1)
A15psd2D = np.abs(A15F2)**2
A15psd1D = radialProfile.azimuthalAverage(A15psd2D)

A16F1 = np.fft.fft2(A16)
A16F2 = np.fft.fftshift(A16F1)
A16psd2D = np.abs(A16F2)**2
A16psd1D = radialProfile.azimuthalAverage(A16psd2D)

A17F1 = np.fft.fft2(A17)
A17F2 = np.fft.fftshift(A17F1)
A17psd2D = np.abs(A17F2)**2
A17psd1D = radialProfile.azimuthalAverage(A17psd2D)

A18F1 = np.fft.fft2(A18)
A18F2 = np.fft.fftshift(A18F1)
A18psd2D = np.abs(A18F2)**2
A18psd1D = radialProfile.azimuthalAverage(A18psd2D)

A19F1 = np.fft.fft2(A19)
A19F2 = np.fft.fftshift(A19F1)
A19psd2D = np.abs(A19F2)**2
A19psd1D = radialProfile.azimuthalAverage(A19psd2D)

A20F1 = np.fft.fft2(A20)
A20F2 = np.fft.fftshift(A20F1)
A20psd2D = np.abs(A20F2)**2
A20psd1D = radialProfile.azimuthalAverage(A20psd2D)






fig = plt.figure()
ax = fig.add_subplot(221)
ax.imshow(np.log10(A1), cmap=py.cm.Greys)
ax.set_title("A1")
ax = fig.add_subplot(222)
ax.imshow(np.log10(A2), cmap=py.cm.Greys)
ax.set_title("A2")
ax = fig.add_subplot(223)
ax.imshow(np.log10(A3), cmap=py.cm.Greys)
ax.set_title("A3")
ax = fig.add_subplot(224)
ax.imshow(np.log10(A4), cmap=py.cm.Greys)
ax.set_title("A4")

fig4 = plt.figure()
ax4 = fig4.add_subplot(221)
ax4.imshow(np.log10(A5), cmap=py.cm.Greys)
ax4.set_title("A5")
ax4 = fig4.add_subplot(222)
ax4.imshow(np.log10(A6), cmap=py.cm.Greys)
ax4.set_title("A6")
ax4 = fig4.add_subplot(223)
ax4.imshow(np.log10(A7), cmap=py.cm.Greys)
ax4.set_title("A7")
ax4 = fig4.add_subplot(224)
ax4.imshow(np.log10(A8), cmap=py.cm.Greys)
ax4.set_title("A8")

fig5 = plt.figure()
ax5 = fig5.add_subplot(221)
ax5.imshow(np.log10(A9), cmap=py.cm.Greys)
ax5.set_title("A9")
ax5 = fig5.add_subplot(222)
ax5.imshow(np.log10(A10), cmap=py.cm.Greys)
ax5.set_title("A10")
ax5 = fig5.add_subplot(223)
ax5.imshow(np.log10(A11), cmap=py.cm.Greys)
ax5.set_title("A11")
ax5 = fig5.add_subplot(224)
ax5.imshow(np.log10(A12), cmap=py.cm.Greys)
ax5.set_title("A12")

fig6 = plt.figure()
ax6 = fig6.add_subplot(221)
ax6.imshow(np.log10(A13), cmap=py.cm.Greys)
ax6.set_title("A13")
ax6 = fig6.add_subplot(222)
ax6.imshow(np.log10(A14), cmap=py.cm.Greys)
ax6.set_title("A14")
ax6 = fig6.add_subplot(223)
ax6.imshow(np.log10(A15), cmap=py.cm.Greys)
ax6.set_title("A15")
ax6 = fig6.add_subplot(224)
ax6.imshow(np.log10(A16), cmap=py.cm.Greys)
ax6.set_title("A16")

fig10 = plt.figure()
ax10 = fig10.add_subplot(221)
ax10.imshow(np.log10(A17), cmap=py.cm.Greys)
ax10.set_title("A17")
ax10 = fig6.add_subplot(222)
ax10.imshow(np.log10(A18), cmap=py.cm.Greys)
ax10.set_title("A18")
ax10 = fig6.add_subplot(223)
ax10.imshow(np.log10(A19), cmap=py.cm.Greys)
ax10.set_title("A19")
ax10 = fig6.add_subplot(224)
ax10.imshow(np.log10(A20), cmap=py.cm.Greys)
ax10.set_title("A20")


###################
fig2 = plt.figure()
ax2 = fig2.add_subplot(221)
ax2.imshow(np.log10(A1psd2D))
ax2.set_title("A1")
ax2 = fig2.add_subplot(222)
ax2.imshow(np.log10(A2psd2D))
ax2.set_title("A2")
ax2 = fig2.add_subplot(223)
ax2.imshow(np.log10(A3psd2D))
ax2.set_title("A3")
ax2 = fig2.add_subplot(224)
ax2.imshow(np.log10(A4psd2D))
ax2.set_title("A4")

fig7 = plt.figure()
ax7 = fig7.add_subplot(221)
ax7.imshow(np.log10(A5psd2D))
ax7.set_title("A5")
ax7 = fig7.add_subplot(222)
ax7.imshow(np.log10(A6psd2D))
ax7.set_title("A6")
ax7 = fig7.add_subplot(223)
ax7.imshow(np.log10(A7psd2D))
ax7.set_title("A7")
ax7 = fig7.add_subplot(224)
ax7.imshow(np.log10(A8psd2D))
ax7.set_title("A8")

fig8 = plt.figure()
ax8 = fig8.add_subplot(221)
ax8.imshow(np.log10(A9psd2D))
ax8.set_title("A9")
ax8 = fig8.add_subplot(222)
ax8.imshow(np.log10(A10psd2D))
ax8.set_title("A10")
ax8 = fig8.add_subplot(223)
ax8.imshow(np.log10(A11psd2D))
ax8.set_title("A11")
ax8 = fig8.add_subplot(224)
ax8.imshow(np.log10(A12psd2D))
ax8.set_title("A12")

fig9 = plt.figure()
ax9 = fig9.add_subplot(221)
ax9.imshow(np.log10(A13psd2D))
ax9.set_title("A13")
ax9 = fig9.add_subplot(222)
ax9.imshow(np.log10(A14psd2D))
ax9.set_title("A14")
ax9 = fig9.add_subplot(223)
ax9.imshow(np.log10(A15psd2D))
ax9.set_title("A15")
ax9 = fig9.add_subplot(224)
ax9.imshow(np.log10(A16psd2D))
ax9.set_title("A16")

fig11 = plt.figure()
ax11 = fig11.add_subplot(221)
ax11.imshow(np.log10(A17psd2D))
ax11.set_title("A17")
ax11 = fig9.add_subplot(222)
ax11.imshow(np.log10(A18psd2D))
ax11.set_title("A18")
ax11 = fig9.add_subplot(223)
ax11.imshow(np.log10(A19psd2D))
ax11.set_title("A19")
ax11 = fig9.add_subplot(224)
ax11.imshow(np.log10(A20psd2D))
ax11.set_title("A20")
############################
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.semilogy(A1psd1D, color = cm.Paired(0.0))
ax3.semilogy(A2psd1D, color = cm.Paired(0.06))
ax3.semilogy(A3psd1D, color = cm.Paired(0.12))
ax3.semilogy(A4psd1D, color = cm.Paired(0.18))
ax3.semilogy(A5psd1D, color = cm.Paired(0.24))
ax3.semilogy(A6psd1D, color = cm.Paired(0.30))
ax3.semilogy(A7psd1D, color = cm.Paired(0.36))
ax3.semilogy(A8psd1D, color = cm.Paired(0.42))
ax3.semilogy(A9psd1D, color = cm.Paired(0.48))
ax3.semilogy(A10psd1D, color = cm.Paired(0.54))
ax3.semilogy(A11psd1D, color = cm.Paired(0.60))
ax3.semilogy(A12psd1D, color = cm.Paired(0.64))
ax3.semilogy(A13psd1D, color = cm.Paired(0.72))
ax3.semilogy(A14psd1D, color = cm.Paired(0.78))
ax3.semilogy(A15psd1D, color = cm.Paired(0.84))
ax3.semilogy(A16psd1D, color = cm.Paired(0.90))
ax3.semilogy(A17psd1D, color = cm.Paired(0.90))
ax3.semilogy(A18psd1D, color = cm.Paired(0.90))
ax3.semilogy(A19psd1D, color = cm.Paired(0.90))
ax3.semilogy(A20psd1D, color = cm.Paired(0.90))


ax3.set_xlabel('Spatial Frequency')
ax3.set_ylabel('Power Spectrum')


psd1Dav = A1psd1D
psd1Dav = np.vstack((psd1Dav, A1psd1D))
psd1Dav = np.vstack((psd1Dav, A2psd1D))
psd1Dav = np.vstack((psd1Dav, A3psd1D))
psd1Dav = np.vstack((psd1Dav, A4psd1D))
psd1Dav = np.vstack((psd1Dav, A5psd1D))
psd1Dav = np.vstack((psd1Dav, A6psd1D))
psd1Dav = np.vstack((psd1Dav, A7psd1D))
psd1Dav = np.vstack((psd1Dav, A8psd1D))
psd1Dav = np.vstack((psd1Dav, A9psd1D))
psd1Dav = np.vstack((psd1Dav, A10psd1D))
psd1Dav = np.vstack((psd1Dav, A11psd1D))
psd1Dav = np.vstack((psd1Dav, A12psd1D))
psd1Dav = np.vstack((psd1Dav, A13psd1D))
psd1Dav = np.vstack((psd1Dav, A14psd1D))
psd1Dav = np.vstack((psd1Dav, A15psd1D))
psd1Dav = np.vstack((psd1Dav, A16psd1D))
psd1Dav = np.vstack((psd1Dav, A17psd1D))
psd1Dav = np.vstack((psd1Dav, A18psd1D))
psd1Dav = np.vstack((psd1Dav, A19psd1D))
psd1Dav = np.vstack((psd1Dav, A20psd1D))


psd1Dav = np.average(psd1Dav, axis=0)

fig10 = plt.figure()
ax10 = fig10.add_subplot(111)
ax10.semilogy(psd1Dav, color='b', linewidth=5.0)
ax10.set_title("Average")



plt.show()



sys.exit()
