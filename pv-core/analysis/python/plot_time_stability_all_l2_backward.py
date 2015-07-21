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

if len(sys.argv) < 3:
   print "usage: time_stability filename l2 post_last filename l2 post"
   print len(sys.argv)
   sys.exit()

w = rw.PVReadWeights(sys.argv[1])

space = 1

d = np.zeros((4,4))

nx = w.nx
ny = w.ny
nxp = w.nxp
nyp = w.nyp
flatnxp = nxp * nyp
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
   ppre = w.file.tell()
   p = w.next_patch()
   ppost = w.file.tell()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         acount = acount + 1
         if kxOn == margin + 1 and kyOn == margin + 1:
            d = p
         else:
            e = p
            d = np.vstack((d,e))
patchblen = ppost - ppre
print patchblen


wd = sp.whiten(d)
result = sp.kmeans2(wd, k)
   
cluster = result[1]
nx_im = 2 * (nxp + space) + space
ny_im = k * (nyp + space) + space

im = np.zeros((nx_im, ny_im))
im[:,:] = (w.max - w.min) / 2.

nx_im2 = nx * (nxp)
ny_im2 = ny * (nyp)

im2 = np.zeros((nx_im2, ny_im2))
im2[:,:] = (w.max - w.min) / 2.

nx_im3 = nx * (nxp)
ny_im3 = ny * (nyp)

im3 = np.zeros((nx_im3, ny_im3))
im3[:,:] = (w.max - w.min) / 2.

  
b = result[0]
con = b

for i in range(k):      
   d = con[i].reshape(nxp, nyp)
   numrows, numcols = d.shape

   x = space + (space + nxp) * (i % k)
   y = space + (space + nyp) * (i / k)

   im[y:y+nyp, x:x+nxp] = d


kcount1 = 0.0
kcount2 = 0.0
kcount3 = 0.0
kcount4 = 0.0
kcount5 = 0.0
kcount6 = 0.0
kcount7 = 0.0
kcount8 = 0.0
kcount9 = 0.0
kcount10 = 0.0
kcount11 = 0.0
kcount12 = 0.0
kcount13 = 0.0
kcount14= 0.0
kcount15 = 0.0
kcount16 = 0.0

for i in range(acount):
   if cluster[i] == 0:
      kcount1 = kcount1 + 1
   if cluster[i] == 1:
      kcount2 = kcount2 + 1
   if cluster[i] == 2:
      kcount3 = kcount3 + 1
   if cluster[i] == 3:
      kcount4 = kcount4 + 1
   if cluster[i] == 4:
      kcount5 = kcount5 + 1
   if cluster[i] == 5:
      kcount6 = kcount6 + 1
   if cluster[i] == 6:
      kcount7 = kcount7 + 1
   if cluster[i] == 7:
      kcount8 = kcount8 + 1
   if cluster[i] == 8:
      kcount9 = kcount9 + 1
   if cluster[i] == 9:
      kcount10 = kcount10 + 1
   if cluster[i] == 10:
      kcount11 = kcount11 + 1
   if cluster[i] == 11:
      kcount12 = kcount12 + 1
   if cluster[i] == 12:
      kcount13 = kcount13 + 1
   if cluster[i] == 13:
      kcount14 = kcount14 + 1
   if cluster[i] == 14:
      kcount15 = kcount15 + 1
   if cluster[i] == 15:
      kcount16 = kcount16 + 1



kcountper1 = kcount1 / acount 
kcountper2 = kcount2 / acount 
kcountper3 = kcount3 / acount 
kcountper4 = kcount4 / acount 
kcountper5 = kcount5 / acount 
kcountper6 = kcount6 / acount 
kcountper7 = kcount7 / acount 
kcountper8 = kcount8 / acount 
kcountper9 = kcount9 / acount 
kcountper10 = kcount10 / acount 
kcountper11 = kcount11 / acount 
kcountper12 = kcount12 / acount 
kcountper13 = kcount13 / acount 
kcountper14 = kcount14 / acount 
kcountper15 = kcount15 / acount 
kcountper16 = kcount16 / acount 




"""
fig = plt.figure()
ax = fig.add_subplot(111)
   
textx = (-7/16.0) * k
texty = (10/16.0) * k
   
ax.set_title('On and Off K-means')
ax.set_axis_off()
ax.text(textx, texty,'ON\n\nOff', fontsize='xx-large', rotation='horizontal') 
ax.text( -5, 12, "Percent %.2f   %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f" %(kcountper1,  kcountper2,  kcountper3, kcountper4, kcountper5, kcountper6, kcountper7, kcountper8, kcountper9, kcountper10, kcountper11, kcountper12, kcountper13, kcountper14, kcountper15, kcountper16), fontsize='large', rotation='horizontal')
ax.text(-4, 14, "Patch   1      2       3       4       5       6       7       8       9      10      11     12     13     14     15     16", fontsize='x-large', rotation='horizontal')

ax.imshow(im, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)

plt.show()
"""
##########
# Choose K-cluster
##########

#feature = input('Please which k-cluster to compare:')

##########
# Find Position of Patches in K-cluster[x]
##########








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
            if cluster[count] == feature:
               e = p
               e = e.reshape(nxp, nyp)
               numrows, numcols = e.shape
               count = count + 1
               patpos = w.file.tell()
               patchposition.append(patpos)
            else:
               e = d
               count = count + 1
         else:
            e = d
      else:
         e = d
      x = (nxp) * (ko % nx)
      y = ( nyp) * (ko / nx)

      im2[y:y+nyp, x:x+nxp] = e

   ##########
   # Find Valuse of K-cluster[x] Patches
   ##########


   w = rw.PVReadWeights(sys.argv[2])
   w.rewind()
   patpla = patchposition
   lenpat = len(patpla)


   number = w.numPatches 
   count = 0

   exp = []
   exppn = []

   body = w.recSize + 4
   hs = w.headerSize
   filesize = os.path.getsize(sys.argv[2])
   bint = filesize / body


   bint = bint - forwardjump - 1

   if forwardjump == 0:
      4
   else:
      leap = ((body * forwardjump) + (100 * forwardjump))
      w.file.seek(leap, os.SEEK_CUR)






   for i in range(bint):
      if i == 0:
         for j in range(lenpat):
            if j == 0:
               go = patpla[0] - hs - patchblen
               w.file.seek(go, os.SEEK_CUR)
               p = w.next_patch()
               if len(p) == 0:
                  print"STOPPEP SUPER  EARLY"
                  sys.exit()
               d = p
               allpat = 0
               fallpat = d


               #p = w.normalize(d)
               #pn = p
               #pn = np.reshape(np.matrix(pn),(1,flatnxp))
               #p = np.reshape(np.matrix(p),(flatnxp,1))
               #pm = pn * p
               #exppn = np.append(exppn, pn)
               #exp = np.append(exp,pm)
               
            else:
               pospost = patpla[j - 1]
               poscur = patpla[j]
               jump = poscur - pospost - patchblen
               w.file.seek(jump, os.SEEK_CUR)
               p = w.next_patch()
               if len(p) == 0:
                  print"STOPPED EARLY"
                  sys.exit()
               d = p
               nallpat = d
               fallpat = np.vstack((fallpat, nallpat))

               #p = w.normalize(d)
               #pn = p
               #pn = np.reshape(np.matrix(pn),(1,flatnxp))
               #p = np.reshape(np.matrix(p),(flatnxp,1))
               #pm = pn * p
               #exppn = np.append(exppn, pn)
               #exp = np.append(exp,pm)
               #print "Ch-Ch-Changes", exppn
      else:
         count = 0
         prejump = body - patpla[lenpat-1] + hs
         w.file.seek(prejump, os.SEEK_CUR)
         for j in range(lenpat):
            if j == 0:
               go = patpla[0] - 4 - patchblen
               w.file.seek(go, os.SEEK_CUR)
               p = w.next_patch()
               test = p
               if len(test) == 0:
                  print "stop"
                  input('Press Enter to Continue')
                  sys.exit()
               d = p
               nfallpat = d


               #p = w.normalize(d)
               #p = np.reshape(np.matrix(p),(flatnxp,1))
               #j1 = 0
               #j2 = flatnxp
               #pm = np.matrix(exppn[j1:j2]) * p
               #exp = np.append(exp,pm)
               #count += 1
            else:
               pospost = patpla[j - 1]
               poscur = patpla[j]
               jump = poscur - pospost - patchblen
               w.file.seek(jump, os.SEEK_CUR)
               p = w.next_patch()
               test = p
               if len(test) == 0:
                  print "stop"
                  input('Press Enter to Continue')
                  sys.exit()
               d = p
               nfallpat = np.vstack((nfallpat, d))

               #p = w.normalize(d)
               #p = np.reshape(np.matrix(p),(flatnxp,1))
               #j1 = flatnxp * j
               #j2 = flatnxp * (j +1)
               #pm = np.matrix(exppn[j1:j2]) * p
               #exp = np.append(exp,pm)
               #count += 1

         fallpat = np.dstack((fallpat, nfallpat))


   ##########
   # Find Average of K-cluster[x] Weights
   ##########

   exp = []
   exppn = []
   dsallpat = np.dsplit(fallpat, bint)
   for i in range(bint):
      postds = dsallpat[-(i+1)]
      sh = np.shape(postds)
      sh = sh[0]
      if i == 0:
         for j in range(sh):
            if j == 0:
               d = postds[j]
               p = w.normalize(d)
               pn = p
               pn = np.reshape(np.matrix(pn), (1,flatnxp))
               p = np.reshape(np.matrix(p), (flatnxp,1))
               pm = pn * p
               exppn = np.append(exppn, pn)
               exp = np.append(exp, pm)
            else:
               d = postds[j]
               p = w.normalize(d)
               pn = p
               pn = np.reshape(np.matrix(pn),(1,flatnxp))
               p = np.reshape(np.matrix(p),(flatnxp,1))
               pm = pn * p
               exppn = np.append(exppn, pn)
               exp = np.append(exp, pm)
      else:
         for j in range(sh):
            if j == 0:
               d = postds[j]
               p = w.normalize(d)
               p = np.reshape(np.matrix(p),(flatnxp,1))
               j1 = 0
               j2 = flatnxp
               pm = np.matrix(exppn[j1:j2]) * p
               exp = np.append(exp, pm)
               count += 1
            else:
               d = postds[j]
               p = w.normalize(d)
               p = np.reshape(np.matrix(p),(flatnxp,1))
               j1 = flatnxp * j
               j2 = flatnxp * (j + 1)
               pm = np.matrix(exppn[j1:j2]) * p
               exp = np.append(exp, pm)
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
            logvalue = patpla[i + (thenumber * j)]
            logvalue = math.log10(logvalue)
            logtotal = np.append(logtotal, logvalue)
         else:
            value = patpla[i + (thenumber * j)]
            subtotal = np.append(subtotal, value) 
            logvalue = patpla[i + (thenumber * j)]
            logvalue = math.log10(logvalue)
            logsubtotal = np.append(logsubtotal, logvalue)
        
      if i > 0:
         total = total + subtotal
      if i > 0:
         logtotal = logtotal + logsubtotal


   total = total / thenumberf
   logtotal = logtotal / thenumberf


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
   global logtotal1
   global logtotal2
   global logtotal3
   global logtotal4
   global logtotal5
   global logtotal6
   global logtotal7
   global logtotal8
   global logtotal9
   global logtotal10
   global logtotal11
   global logtotal12
   global logtotal13
   global logtotal14
   global logtotal15
   global logtotal16

   if feature == 0:
      total1 = [0.0]
      total2 = [0.0]
      total3 = [0.0]
      total4 = [0.0]
      total5 = [0.0]
      total6 = [0.0]
      total7 = [0.0]
      total8 = [0.0]
      total9 = [0.0]
      total10 = [0.0]
      total11 = [0.0]
      total12 = [0.0]
      total13 = [0.0]
      total14 = [0.0]
      total15 = [0.0]
      total16 = [0.0]
      logtotal1 = [0.0]
      logtotal2 = [0.0]
      logtotal3 = [0.0]
      logtotal4 = [0.0]
      logtotal5 = [0.0]
      logtotal6 = [0.0]
      logtotal7 = [0.0]
      logtotal8 = [0.0]
      logtotal9 = [0.0]
      logtotal10 = [0.0]
      logtotal11 = [0.0]
      logtotal12 = [0.0]
      logtotal13 = [0.0]
      logtotal14 = [0.0]
      logtotal15 = [0.0]
      logtotal16 = [0.0]

   if feature == 0:
      total1 = total 
      logtotal1 = logtotal
   if feature == 1:
      total2 = total
      logtotal2 = logtotal
   if feature == 2:
      total3 = total
      logtotal3 = logtotal
   if feature == 3:
      total4 = total
      logtotal4 = logtotal
   if feature == 4:
      total5 = total
      logtotal5 = logtotal
   if feature == 5:
      total6 = total
      logtotal6 = logtotal
   if feature == 6:
      total7 = total
      logtotal7 = logtotal
   if feature == 7:
      total8 = total
      logtotal8 = logtotal
   if feature == 8:
      total9 = total
      logtotal9 = logtotal
   if feature == 9:
      total10 = total
      logtotal10 = logtotal
   if feature == 10:
      total11 = total
      logtotal11 = logtotal
   if feature == 11:
      total12 = total
      logtotal12 = logtotal
   if feature == 12:
      total13 = total
      logtotal13 = logtotal
   if feature == 13:
      total14 = total
      logtotal14 = logtotal
   if feature == 14:
      total15 = total
      logtotal15 = logtotal
   if feature == 15:
      total16 = total
      logtotal16 = logtotal

   return





w = rw.PVReadWeights(sys.argv[2])

body = w.recSize + 4
hs = w.headerSize
filesize = os.path.getsize(sys.argv[2])
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
ax2.plot(np.arange(len(logtotal1)), logtotal1, '-o', color='y')
ax2.plot(np.arange(len(total2)), total2, '-o', color='r')
ax2.plot(np.arange(len(logtotal2)), logtotal2, '-o', color='r')
ax2.plot(np.arange(len(total3)), total3, '-o', color='b')
ax2.plot(np.arange(len(logtotal3)), logtotal3, '-o', color='b')
ax2.plot(np.arange(len(total4)), total4, '-o', color='c')
ax2.plot(np.arange(len(logtotal4)), logtotal4, '-o', color='c')
ax2.plot(np.arange(len(total5)), total5, '-o', color='m')
ax2.plot(np.arange(len(logtotal5)), logtotal5, '-o', color='m')
ax2.plot(np.arange(len(total6)), total6, '-o', color='k')
ax2.plot(np.arange(len(logtotal6)), logtotal6, '-o', color='k')
ax2.plot(np.arange(len(total7)), total7, '-o', color='w')
ax2.plot(np.arange(len(logtotal7)), logtotal7, '-o', color='w')
ax2.plot(np.arange(len(total8)), total8, '-o', color='g')
ax2.plot(np.arange(len(logtotal8)), logtotal8, '-o', color='g')

print "yellow = 1, 9"
print "red = 2, 10"
print "blue = 3, 11"
print "cyan = 4, 12"
print "magenta = 5, 13"
print "black = 6, 14"
print "white = 7, 15"
print "green = 8, 16"


ax3.plot(np.arange(len(total9)), total9, '-o', color='y')
ax3.plot(np.arange(len(logtotal9)), logtotal9, '-o', color='y')
ax3.plot(np.arange(len(total10)), total10, '-o', color='r')
ax3.plot(np.arange(len(logtotal10)), logtotal10, '-o', color='r')
ax3.plot(np.arange(len(total11)), total11, '-o', color='b')
ax3.plot(np.arange(len(logtotal11)), logtotal11, '-o', color='b')
ax3.plot(np.arange(len(total12)), total12, '-o', color='c')
ax3.plot(np.arange(len(logtotal12)), logtotal12, '-o', color='c')
ax3.plot(np.arange(len(total13)), total13, '-o', color='m')
ax3.plot(np.arange(len(logtotal13)), logtotal13, '-o', color='m')
ax3.plot(np.arange(len(total14)), total14, '-o', color='k')
ax3.plot(np.arange(len(logtotal14)), logtotal14, '-o', color='k')
ax3.plot(np.arange(len(total15)), total15, '-o', color='w')
ax3.plot(np.arange(len(logtotal15)), logtotal15, '-o', color='w')
ax3.plot(np.arange(len(total16)), total16, '-o', color='g')
ax3.plot(np.arange(len(logtotal16)), logtotal16, '-o', color='g')


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
