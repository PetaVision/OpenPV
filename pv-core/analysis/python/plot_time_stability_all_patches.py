"""
Plots the time stability
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import PVReadWeights as rw
import PVConversions as conv
import scipy.cluster.vq as sp
import math

if len(sys.argv) < 5:
   print "usage: time_stability filename on, filename off, filename-on post, filename-off post"
   print len(sys.argv)
   sys.exit()

w = rw.PVReadWeights(sys.argv[1])
wOff = rw.PVReadWeights(sys.argv[2])

space = 1

d = np.zeros((4,4))

nx = w.nx
ny = w.ny
nxp = w.nxp
nyp = w.nyp
numpat = w.numPatches
nf = w.nf
margin = 10
marginstart = margin
marginend = nx - margin
acount = 0
patchposition = []


def format_coord(x, y):
   col = int(x+0.5)
   row = int(y+0.5)
   x2 = (x / 16.0)
   y2 = (y / 16.0) 
   x = (x / 4.0)
   y = (y / 4.0)
   if col>=0 and col<numcols and row>=0 and row<numrows:
      z = P[row,col]
      return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
   else:
      return 'x=%1.4d, y=%1.4d, x2=%1.4d, y2=%1.4d'%(int(x), int(y), int(x2), int(y2))



k = 16
for ko in range(numpat):
   kxOn = conv.kxPos(ko, nx, ny, nf)
   kyOn = conv.kyPos(ko, nx, ny, nf)
   p = w.next_patch()
   poff = wOff.next_patch()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         acount = acount + 1
         if kxOn == margin + 1 and kyOn == margin + 1:
            





         else:





total = []
logtotal = []

def k_stability_analysis(k, forwardjump):
   w = rw.PVReadWeights(sys.argv[1])
   feature = k - 1
   count = 0
   d = np.zeros((nxp,nyp))

   w.rewind()
   for ko in np.arange(numpat):
      kxOn = conv.kxPos(ko, nx, ny, nf)
      kyOn = conv.kyPos(ko, nx, ny, nf)
      p = w.next_patch()
      if marginstart < kxOn < marginend:
         if marginstart < kyOn < marginend:





   ##########
   # Find Valuse of K-cluster[x] Patches
   ##########


   w = rw.PVReadWeights(sys.argv[3])
   wOff = rw.PVReadWeights(sys.argv[4])
   w.rewind()
   wOff.rewind()
   patpla = patchposition
   lenpat = len(patpla)


   number = w.numPatches 
   count = 0

   exp = []
   expOff = []
   exppn = []
   exppnOff = []

   body = w.recSize + 4
   hs = w.headerSize
   filesize = os.path.getsize(sys.argv[3])
   bint = filesize / body


   bint = bint - forwardjump - 1

   if forwardjump == 0:
      print "43110"
   else:
      leap = (body * forwardjump)
      w.file.seek(leap, os.SEEK_CUR)






   for i in range(bint):
      if i == 0:
         for j in range(lenpat):
            if j == 0:
               p = w.next_patch()
               pOff = wOff.next_patch()
               if len(p) == 0:
                  print"STOPPEP SUPER  EARLY"
                  sys.exit()
               don = p
               doff = pOff
               d = np.append(don, doff)
               p = w.normalize(d)
               pn = p
               pn = np.reshape(np.matrix(pn),(1,32))
               p = np.reshape(np.matrix(p),(32,1))
               pm = pn * p
               exppn = np.append(exppn, pn)
               exp = np.append(exp,pm)
            else:
               p = w.next_patch()
               pOff = wOff.next_patch()
               if len(pOff) == 0:
                  print"STOPPED EARLY"
                  sys.exit()
               don = p
               doff = pOff
               d = np.append(don, doff)
               p = w.normalize(d)
               pn = p
               pn = np.reshape(np.matrix(pn),(1,32))
               p = np.reshape(np.matrix(p),(32,1))
               pm = pn * p
               exppn = np.append(exppn, pn)
               exp = np.append(exp,pm)
      else:
         count = 0
         prejump = body - patpla[lenpat-1] + hs
         w.file.seek(prejump, os.SEEK_CUR)
         wOff.file.seek(prejump, os.SEEK_CUR)
         for j in range(lenpat):
            if j == 0:
               p = w.next_patch()
               pOff = wOff.next_patch()
               test = p
               if len(test) == 0:
                  print "stop"
                  input('Press Enter to Continue')
                  sys.exit()
               don = p
               doff = pOff
               d = np.append(don, doff)
               p = w.normalize(d)
               p = np.reshape(np.matrix(p),(32,1))
               j1 = 0
               j2 = 32
               pm = np.matrix(exppn[j1:j2]) * p
               exp = np.append(exp,pm)
               count += 1
            else:
               p = w.next_patch()
               pOff = wOff.next_patch()
               test = pOff
               if len(test) == 0:
                  print "stop"
                  input('Press Enter to Continue')
                  sys.exit()
               don = p
               doff = pOff
               d = np.append(don, doff)
               p = w.normalize(d)
               p = np.reshape(np.matrix(p),(32,1))
               j1 = 32 * j
               j2 = 32 * (j +1)
               pm = np.matrix(exppn[j1:j2]) * p
               exp = np.append(exp,pm)
               count += 1



   ##########
   # Find Average of K-cluster[x] Weights
   ##########


   thenumber = lenpat
   thenumberf = float(thenumber)

   patpla = exp
   lenpat = len(patpla)


   howlong = lenpat / thenumber

   total = []
   logtotal = []

   for i in range(thenumber):
      subtotal = []
      logsubtotal = []
      for j in range(howlong):
         if i == 0:
            value = patpla[i + (thenumber * j)]
            total = np.append(total, value)
            #logvalue = patpla[i + (thenumber * j)]
            #logvalue = math.log10(logvalue)
            #logtotal = np.append(logtotal, logvalue)
         else:
            value = patpla[i + (thenumber * j)]
            subtotal = np.append(subtotal, value) 
            #logvalue = patpla[i + (thenumber * j)]
            #logvalue = math.log10(logvalue)
            #logsubtotal = np.append(logsubtotal, logvalue)
        
      if i > 0:
         total = total + subtotal
      #if i > 0:
         #logtotal = logtotal + logsubtotal


   total = total / thenumberf
   #logtotal = logtotal / thenumberf


   global total1
   global total2
   global total3
   global total4
   global total5
   global total6
   global total7
   global total8
   global total9
   global total10
   global total11
   global total12
   global total13
   global total14
   global total15
   global total16




   #global logtotal1
   #global logtotal2
   #global logtotal3
   #global logtotal4
   #global logtotal5
   #global logtotal6
   #global logtotal7
   #global logtotal8
   #global logtotal9
   #global logtotal10
   #global logtotal11
   #global logtotal12
   #global logtotal13
   #global logtotal14
   #global logtotal15
   #global logtotal16

   if feature == 0:
      total1 = total 
   if feature == 1:
      total2 = total
   if feature == 2:
      total3 = total
   if feature == 3:
      total4 = total
   if feature == 4:
      total5 = total
   if feature == 5:
      total6 = total
   if feature == 6:
      total7 = total
   if feature == 7:
      total8 = total
   if feature == 8:
      total9 = total
   if feature == 9:
      total10 = total
   if feature == 10:
      total11 = total
   if feature == 11:
      total12 = total
   if feature == 12:
      total13 = total
   if feature == 13:
      total14 = total
   if feature == 14:
      total15 = total
   if feature == 15:
      total16 = total

   return





w = rw.PVReadWeights(sys.argv[3])

body = w.recSize + 4
hs = w.headerSize
filesize = os.path.getsize(sys.argv[3])
bint = filesize / body

print
print "Number of steps = ", bint
forwardjump = input('How many steps forward:')


count = 0

for i in range(16):
   i = i + 1
   k_stability_analysis(i, forwardjump)
   count += 1
   print count

if len(total1) == 0:
   total1 = .5
if len(total2) == 0:
   total2 = .5
if len(total3) == 0:
   total3 = .5
if len(total4) == 0:
   total4 = .5
if len(total5) == 0:
   total5 = .5
if len(total6) == 0:
   total6 = .5
if len(total7) == 0:
   total7 = .5
if len(total8) == 0:
   total8 = .5
if len(total9) == 0:
   total9 = .5
if len(total10) == 0:
   total10 = .5
if len(total11) == 0:
   total11 = .5
if len(total12) == 0:
   total12 = .5
if len(total13) == 0:
   total13 = .5
if len(total14) == 0:
   total14 = .5
if len(total15) == 0:
   total15 = .5
if len(total16) == 0:
   total16 = .5



##########
# Plot Time Stability Curve
##########



fig = plt.figure()
ax = fig.add_subplot(111)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, axisbg='darkslategray')
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, axisbg='darkslategray')


textx = (-7/16.0) * k
texty = (10/16.0) * k
   
ax.set_title('On and Off K-means')
ax.set_axis_off()
ax.text(textx, texty,'ON\n\nOff', fontsize='xx-large', rotation='horizontal') 
ax.text( -5, 12, "Percent %.2f   %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f" %(kcountper1,  kcountper2,  kcountper3, kcountper4, kcountper5, kcountper6, kcountper7, kcountper8, kcountper9, kcountper10, kcountper11, kcountper12, kcountper13, kcountper14, kcountper15, kcountper16), fontsize='large', rotation='horizontal')
ax.text(-4, 14, "Patch   1      2       3       4       5       6       7       8       9      10      11     12     13     14     15     16", fontsize='x-large', rotation='horizontal')
ax.imshow(im, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)


ax2.plot(np.arange(len(total1)), total1, '-o', color='y')
ax2.plot(np.arange(len(total2)), total2, '-o', color='r')
ax2.plot(np.arange(len(total3)), total3, '-o', color='b')
ax2.plot(np.arange(len(total4)), total4, '-o', color='c')
ax2.plot(np.arange(len(total5)), total5, '-o', color='m')
ax2.plot(np.arange(len(total6)), total6, '-o', color='k')
ax2.plot(np.arange(len(total7)), total7, '-o', color='w')
ax2.plot(np.arange(len(total8)), total8, '-o', color='g')

print "yellow = 1, 9"
print "red = 2, 10"
print "blue = 3, 11"
print "cyan = 4, 12"
print "magenta = 5, 13"
print "black = 6, 14"
print "white = 7, 15"
print "green = 8, 16"


ax3.plot(np.arange(len(total9)), total9, '-o', color='y')
ax3.plot(np.arange(len(total10)), total10, '-o', color='r')
ax3.plot(np.arange(len(total11)), total11, '-o', color='b')
ax3.plot(np.arange(len(total12)), total12, '-o', color='c')
ax3.plot(np.arange(len(total13)), total13, '-o', color='m')
ax3.plot(np.arange(len(total14)), total14, '-o', color='k')
ax3.plot(np.arange(len(total15)), total15, '-o', color='w')
ax3.plot(np.arange(len(total16)), total16, '-o', color='g')


ax2.set_xlabel('Time')
ax2.set_ylabel('Avg Correlation')
ax2.set_title('Time Stability k 1-8')
ax2.set_xlim(0, len(total1))
ax2.grid(True)

ax3.set_xlabel('Time')
ax3.set_ylabel('Avg Correlation')
ax3.set_title('Time Stability k 9-16')
ax3.set_xlim(0, len(total1))
ax3.grid(True)

plt.show()
