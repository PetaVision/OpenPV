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

if len(sys.argv) < 3:
   print "usage: plot_correlations filename On, Off"
   print len(sys.argv)
   sys.exit()

w = rw.PVReadWeights(sys.argv[1])
wOff = rw.PVReadWeights(sys.argv[2])

nxp = w.nxp

# create feature list for comparing weights from on and off cells
f = np.zeros(w.patchSize)
f2 = np.zeros(w.patchSize)
fOn = []
fOff = []
f = w.normalize(f)
f2 = w.normalize(f2)

# vertical lines from right side
#f = np.zeros([w.nxp, w.nyp]) # first line
#f[:,0] = 1
#fOn.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[:,1] = 1
#fOff.append(f)

#this was in use
#f = np.zeros([w.nxp, w.nyp]) # second line
#f[:,1] = 1
#fOn.append(f)
#f = np.zeros([w.nxp, w.nyp])
#f[:,2] = 1
#fOff.append(f)

#f2 = np.zeros([w.nxp, w.nyp]) # third line
#f2[:,2] = 1
#fOn.append(f2)
#f2 = np.zeros([w.nxp, w.nyp])
#f2[:,3] = 1
#fOff.append(f2)

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
f = np.zeros([w.nxp, w.nyp])
f[1,:] = 1
fOn.append(f)
f = np.zeros([w.nxp, w.nyp])
f[2,:] = 1
fOff.append(f)



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
fOn = fOn[0]
fOff = fOff[0]
corrMatrixOn = np.zeros([4,4])
corrMatrixOff = np.zeros([4,4])
corrMatrixOffaOn = np.zeros([4,4])
countOn = 0
countOff = 0
countOffaOn = 0
dWMax = 0.5
realmax = nxp * dWMax


x = w.numPatches
nx = w.nx
ny = w.ny
nf = w.nf
nxp = w.nxp
nyp = w.nyp
margin = 10
marginstart = margin
marginend = nx - margin
n = w.numPatches - (4 * (margin * (nx - margin)))
po = 0.75 #Percent of realmax

for k in np.arange(x):
   pOn = w.next_patch()
   pOff = wOff.next_patch()
   kx = conv.kxPos(k, nx, ny, nf)
   ky = conv.kyPos(k, nx, ny, nf)
   if marginstart < kx < marginend:
      if marginstart < ky < marginend:
         pOn = pOn.reshape(nxp,nyp)
         pOff = pOff.reshape(nxp,nyp)
         resOn = np.sum(pOn * fOn)
         resOff = np.sum(pOff * fOff)
         if resOff >= po * realmax:
            corrMatrixOff = corrMatrixOff + pOff
            countOff += 1
         if resOn >= po * realmax:
            corrMatrixOn = corrMatrixOn + pOn
            countOn += 1
            if resOff >= po * realmax:
               countOffaOn += 1
               corrMatrixOffaOn = corrMatrixOffaOn + pOff              

#         j = 1290 +(g + 1) + (h * 128)

corrMatrixOn = corrMatrixOn / countOn
corrMatrixOff = corrMatrixOff / countOff
corrMatrixOffaOn = corrMatrixOffaOn / countOffaOn
print "fOn                   fOff"
print fOn[0],"   ", fOff[0] 
print fOn[1],"   ", fOff[1] 
print fOn[2],"   ", fOff[2]
print fOn[3],"   ", fOff[3]
print
print "On"
print corrMatrixOn
print "Off"
print corrMatrixOff
print
print "Off next to On"
print corrMatrixOffaOn
print
print "number of cells = ", n
print
print "count of on cells  = ", countOn
print "count of off cells = ", countOff
print "count of off cells next to on = ", countOffaOn
print 
countOn = float(countOn)
print "Percent of on cells = ", (countOn / n) * 100
countOff = float(countOff)
print "Percent of off cells = ", (countOff / n) * 100
countOffaOn = float(countOffaOn)
print "Percent of off cells next to on = ", (countOffaOn / n) * 100
print
#print "list max = ",lstmax
#print "realmax = ", realmax

sys.exit()
