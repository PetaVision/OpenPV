"""
Make a histogram of normally distributed random numbers and plot the
analytic PDF over it
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import PVReadWeights2 as rw

space = 1

w = rw.PVReadWeights(sys.argv[1])


nx  = w.nx
ny  = w.ny
nxp = w.nxp
nyp = w.nyp
print nxp * nyp
print nxp, nyp
nfp = w.nfp
numParams = w.numParams
wmin = w.min
wmax = w.max
params = w.params
numpat = w.numPatches

print "wmax = ", wmax
print wmin
#sys.exit()

for i in range(numpat):
   p = w.next_patch()
   nxny = p[0]
   p = p[1]
   #print nxny
   #print p
   #print
   #if i == 4:
   #   sys.exit()
   if i == 0:
      pat = p
      lan = len(p)
      xy = nxny
   else:
      pat = np.append(pat, p)
      lan = np.append(lan, len(p))
      xy = np.vstack((xy, nxny))


"""
a = np.fromfile(sys.argv[1], 'i', 25)
f = open('test.pvp', 'wb')
f.write(a)
f.write(pat)
print 'fin'
sys.exit()
"""
avg = np.average(pat)
shape = np.shape(pat)

mx = np.max(pat)

for i in range(len(pat)):
   if pat[i] < (wmax / 2.0):
      pat[i] = 0.0


#print "pat = ", pat
#print "shape = ", np.shape(pat)
total = nxp * nyp * nfp


print
print "params = ", params
print type(params)

group = []

for i in range(len(pat)): 
   p = pat[i]
   p = (p - wmin) / (wmax - wmin)
   p = p * 255
   group = np.append(group, p)
   #p.tofile("test.pvp", 'np.uint8()')




e = open(sys.argv[1])

e.seek(0)
head = np.fromfile(e, 'i', 3)
h1 = head[1]-8
print "h1 = ", h1
e.seek(0)
par = np.fromfile(e, 'i', h1)
time = np.fromfile(e, 'd', 1)
xyf = np.fromfile(e, 'i', 3)
minmax = np.fromfile(e, 'f', 2)
nupa = np.fromfile(e, 'i', 1)


print "begin writing process..."

f = open('test.pvp', 'wb')
f.write(par)
f.write(time)
f.write(xyf)
f.write(minmax)
f.write(nupa)

count = 0
for i in range(numpat):
   f.write(np.int16(xy[i][0]))
   f.write(np.int16(xy[i][1]))
   for h in range(lan[i]):
      #print "i = ", i
      #print "count = ", count
      #print len(group)
      #print "group = ", group[count]
      f.write(np.uint8(group[count]))

      count+=1
      if lan[i]-1 == h:
         for i in range(nxp*nyp - lan[i]):
            f.write(np.uint8(0))


#params.tofile("test.pvp", format='np.uint8')
#params.tofile("test.txt", '/n')

print params

print "fin"
