"""
Plot a color map of features of given phase
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import PVReadWeights as rw
import PVConversions as conv

def format_coord(x, y):
    col = int(x+0.5)
    row = int(y+0.5)
    if col>=0 and col<numcols and row>=0 and row<numrows:
        z = M[row,col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f'%(x, y)

"""
Show how to modify the coordinate formatter to report the image "z"
value of the nearest pixel given x and y
"""

if len(sys.argv) < 4:
   print "usage: plot_correlations filename orientation ('vertical' or 'horizontal')"
   print len(sys.argv)
   sys.exit()

w = rw.PVReadWeights(sys.argv[1])
wOff = rw.PVReadWeights(sys.argv[2])

# create feature list for comparing weights from on and off cells
#
f = np.zeros(w.patchSize)
fOn = []
fOff = []

f = w.normalize(f)

# vertical lines from right side
#f = np.zeros([w.nxp, w.nyp]) # first line
#f[:,0] = 1
#fOn.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[:,1] = 1
#fOff.append(f)

#this was in use
f = np.zeros([w.nxp, w.nyp]) # second line
f[:,1] = 1
fOn.append(f)
f = np.zeros([w.nxp, w.nyp])
f[:,2] = 1
fOff.append(f)

#f = np.zeros([w.nxp, w.nyp]) # third line
#f[:,2] = 1
#fOn.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[:,3] = 1
#fOff.append(f)

#vertical lines from left side
#f = np.zeros([w.nxp, w.nyp])
#f[:,3] = 1
#fOn.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[:,2] = 1
#fOff.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[:,2] = 1
#fOn.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[:,1] = 1
#fOff.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[:,1] = 1
#fOn.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[:,0] = 1
#fOff.append(f)

#horizontal lines from the bottom
#f = np.zeros([w.nxp, w.nyp])
#f[0,:] = 1
#fOn.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[1,:] = 1
#fOff.append(f)


#THIS WAS USED
#f = np.zeros([w.nxp, w.nyp])
#f[1,:] = 1
#fOn.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[2,:] = 1
#fOff.append(f)



#f = np.zeros([w.nxp, w.nyp])
#f[2,:] = 1
#fOn.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[3,:] = 1
#fOff.append(f)

#horizontal lines from the top
#f = np.zeros([w.nxp, w.nyp])
#f[3,:] = 1
#fOn.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[2,:] = 1
#fOff.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[2,:] = 1
#fOn.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[1,:] = 1
#fOff.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[1,:] = 1
#fOn.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[0,:] = 1
#fOff.append(f)

#Corners starting top right and going clockwise
#f = np.zeros([w.nxp, w.nyp])
#f[0,:] = 1
#f[:,0] = 1
#fOn.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[1:4, 1:2] = 1
#f[1:2, 2:4] = 1
#fOff.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[0,:] = 1
#f[:,3] = 1
#fOn.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[1:2, 0:3] = 1
#f[2:4, 2:3] = 1
#fOff.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[3,:] = 1
#f[:,3] = 1
#fOn.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[2:3, 0:3] = 1
#f[0:2, 2:3] = 1
#fOff.append(f)
#f = np.zeros([w.nxp, w.nyp])
#t[3,:] = 1
#t[:,0] = 1
#fOn.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[0:3, 1:2] = 1
#f[2:3, 2:4] = 1
#fOff.append(f)

lst = []
clst = []
f = fOn[0]
corrMatrixOn = np.zeros([4,4])
corrMatrixOff = np.zeros([4,4])
for i in range(w.numPatches):
   p = w.next_patch()
   p = p.reshape(4,4)
   res = np.sum(p * f)
   if res > 0:
      lst.append(res)

lstmax = max(lst)

w.rewind()

count = 0
for i in range(w.numPatches):
   p = w.next_patch()
   p = p.reshape(4,4)
   pOff = wOff.next_patch()
   res = np.sum(p * f)
   if res > 0.5 * lstmax:
      count += 1
      pOff = pOff.reshape(4,4)
      corrMatrixOn = corrMatrixOn + p
      corrMatrixOff = corrMatrixOff + pOff

corrMatrixOn = corrMatrixOn / count
corrMatrixOff = corrMatrixOff / count
print "count = ", count
print f
print "On"
print corrMatrixOn
print "Off"
print corrMatrixOff
sys.exit()

for i in range(wOff.numPatches):
   p = wOff.next_patch()
   p = p.reshape(4,4)
   res = np.sum(p*f)




print
print
print fOn
print
print
print fOff
print
print

sys.exit()


# write loop to normalize all features in the on and off lists


numrows = w.nyGlobal
numcols = w.nxGlobal

M = np.zeros(w.numPatches)

for k in range(len(M)):
   p = w.normalize( w.next_patch() )
   M[k] = np.sum(p * f)

print "time =", w.time


# calculate distributions
#
numbins = 101
dist = np.zeros(numbins)
bins = np.zeros(numbins)
count = numrows * numcols
for k in range(numbins): bins[k] = k * 1.0/(numbins-1)
for k in range(len(M)):
    for b in range(numbins):
       if (M[k] > bins[b]): dist[b] += 1
dist = dist/count

# print maximum projection
#
maxp = 0
maxk = 0
for k in range(len(M)):
   if M[k] > maxp: maxp = M[k]; maxk = k

kx = conv.kxPos(k, numcols, numrows, w.nf)
ky = conv.kyPos(k, numcols, numrows, w.nf)
print "maximum projected value = ", maxp, maxk, kx, ky

M = M.reshape( (numrows,numcols) )

# print averaged projection over column
#
s = np.zeros(numcols)
maxs = 0.0
maxcol = 0
for col in range(numcols):
   s[col] = np.sum(M[:,col])
   if s[col] > maxs: maxs = s[col]; maxcol = col
s = s/numrows
print "(maxcol, maxsum) = (", maxcol, ",", maxs/numrows, ")"

fig = plt.figure()

ax = fig.add_subplot(2,1,1)
ax.set_xlabel('Kx GLOBAL')
ax.set_ylabel('Ky GLOBAL')
ax.set_title( 'Feature Projection: phase=%d' %(phase) )
ax.format_coord = format_coord
ax.imshow(M, cmap=cm.jet, interpolation='nearest', vmin=0., vmax=1.)

ax = fig.add_subplot(2,1,2)
ax.set_xlabel('Feature Strength')
ax.set_ylabel('Percentage')
ax.plot(bins, dist, 'o', color='blue')
#ax.set_ylim(0.0, 0.5)

plt.show()
