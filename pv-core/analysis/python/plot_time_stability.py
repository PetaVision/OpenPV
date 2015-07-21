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
   print "usage: time_stability filename on, filename off, filename_post"
   print len(sys.argv)
   sys.exit()

w = rw.PVReadWeights(sys.argv[1])


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
supereasytest = 1


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


##########
# Make K-clusters
##########

k = 8
for ko in range(numpat):
   kxOn = conv.kxPos(ko, nx, ny, nf)
   kyOn = conv.kyPos(ko, nx, ny, nf)
   p = w.next_patch()
   if marginstart < kxOn < marginend:
      if marginstart < kyOn < marginend:
         acount = acount + 1
         if kxOn == margin + 1 and kyOn == margin + 1:
            d = p
         else:
            d = np.vstack((d,p))


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




kcountper1 = kcount1 / acount 
kcountper2 = kcount2 / acount 
kcountper3 = kcount3 / acount 
kcountper4 = kcount4 / acount 
kcountper5 = kcount5 / acount 
kcountper6 = kcount6 / acount 
kcountper7 = kcount7 / acount 
kcountper8 = kcount8 / acount 





fig = plt.figure()
ax = fig.add_subplot(111)
   
textx = (-7/16.0) * k
texty = (10/16.0) * k
   
ax.set_title('On and Off K-means')
ax.set_axis_off()
ax.text(textx, texty,'ON\n\nOff', fontsize='xx-large', rotation='horizontal') 
ax.text( -5, 12, "Percent %.2f   %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f" %(kcountper1,  kcountper2,  kcountper3, kcountper4, kcountper5, kcountper6, kcountper7, kcountper8), fontsize='large', rotation='horizontal')
ax.text(-4, 14, "Patch   1      2       3       4       5       6       7       8", fontsize='x-large', rotation='horizontal')

ax.imshow(im, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)

plt.show()

##########
# Choose K-cluster
##########

feature = input('Please which k-cluster to compare:')

##########
# Find Position of Patches in K-cluster[x]
##########

count = 0
d = np.zeros((nxp,nyp))
feature = feature - 1

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
            #print patpos
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
# Plot K-cluster[x] Patches
##########

"""
fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('Kx GLOBAL')
ax.set_ylabel('Ky GLOBAL')
ax.set_title('ON Weight Patches')
ax.format_coord = format_coord
ax.imshow(im2, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)
   

plt.show()   
"""

###  TIME STABILITY PART STARTS HERE


##########
# Find Valuse of K-cluster[x] Patches
##########


w = rw.PVReadWeights(sys.argv[2])
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

print
print "Number of steps = ", bint
forwardjump = input('How many steps forward:')

bint = bint - forwardjump

if forwardjump == 0:
   43110
else:
      leap = (body * forwardjump)
      w.file.seek(leap, os.SEEK_CUR)



countso = 0

for i in range(bint):
   countso+=1
   print countso
   if i == 0:
      for j in range(lenpat):
         if j == 0:
            go = patpla[0] - hs - 20
            w.file.seek(go, os.SEEK_CUR)
            p = w.next_patch()
            if len(p) == 0:
               print "Stopped Super Early"
               sys.exit()
            p = w.normalize(p)
            pn = p
            pn = np.reshape(np.matrix(pn),(1,16))
            p = np.reshape(np.matrix(p),(16,1))
            pm = pn * p
            exppn = np.append(exppn, pn)
            exp = np.append(exp,pm)
         else:
            pospost = patpla[j - 1]
            poscur = patpla[j]
            jump = poscur - pospost - 20
            w.file.seek(jump, os.SEEK_CUR)
            p = w.next_patch()
            if len(p) == 0:
               print"Stopped Early"
               sys.exit()
            p = w.normalize(p)
            pn = p
            pn = np.reshape(np.matrix(pn),(1,16))
            p = np.reshape(np.matrix(p),(16,1))
            pm = pn * p
            exppn = np.append(exppn, pn)
            exp = np.append(exp,pm)
   else:
      count = 0
      prejump = body - patpla[lenpat-1] + hs
      w.file.seek(prejump, os.SEEK_CUR)
      for j in range(lenpat):
         if j == 0:
            go = patpla[0] - hs - 20
            w.file.seek(go, os.SEEK_CUR)
            p = w.next_patch()
            test = p
            if len(test) == 0:
               print "Stop"
               input('Press Enter to Continue')
               sys.exit()
            p = w.normalize(p)
            p = np.reshape(np.matrix(p),(16,1))
            j1 = 0
            j2 = 16
            pm = np.matrix(exppn[j1:j2]) * p
            exp = np.append(exp,pm)
            count += 1
         else:
            pospost = patpla[j - 1]
            poscur = patpla[j]
            jump = poscur - pospost - 20
            w.file.seek(jump, os.SEEK_CUR)
            p = w.next_patch()
            test = p
            if len(test) == 0:
               print "stop"
               input('Press Enter to Continue')
               sys.exit()
            p = w.normalize(p)
            p = np.reshape(np.matrix(p),(16,1))
            j1 = 16 * j
            j2 = 16 * (j +1)
            pm = np.matrix(exppn[j1:j2]) * p
            exp = np.append(exp,pm)
            count += 1

print "THE END"


#############################################



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
#         logvalue = patpla[i + (thenumber * j)]
#         logvalue = math.log10(logvalue)
#         logtotal = np.append(logtotal, logvalue)
      else:
         value = patpla[i + (thenumber * j)]
         subtotal = np.append(subtotal, value) 
#         logvalue = patpla[i + (thenumber * j)]
#         logvalue = math.log10(logvalue)
#         logsubtotal = np.append(logsubtotal, logvalue)
        
   if i > 0:
      total = total + subtotal
#   if i > 0:
#      logtotal = logtotal + logsubtotal



total = total / thenumberf
#logtotal = logtotal / thenumberf
print len(total)


##########
# Plot Time Stability Curve
##########

fig = plt.figure()
ax = fig.add_subplot(111, axisbg='darkslategray')

ax.plot(np.arange(len(total)), total, '-o', color='y')

ax.set_xlabel('Time')
ax.set_ylabel('Avg Correlation')
ax.set_title('Time Stability')
ax.set_xlim(0, len(total))
ax.grid(True)


#ax2 = fig.add_subplot(111, axisbg='darkslategray')
#ax2.plot(np.arange(len(logtotal)), logtotal, '-o', color='y')
#ax2.set_xlabel('Time')
#ax2.set_ylabel('Log of Avg Correlation')
#ax2.set_title('Time Stability')
#ax2.set_xlim(0, len(logtotal))
#ax2.grid(True)



plt.show()
