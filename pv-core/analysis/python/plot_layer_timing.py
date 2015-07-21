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
from scipy.sparse import lil_matrix as lm


extended = False

l1a = rs.PVReadSparse(sys.argv[1], extended)
l1w = rw.PVReadWeights(sys.argv[2])
l2a = rs.PVReadSparse(sys.argv[3], extended)
l2w = rw.PVReadWeights(sys.argv[4])
l3a = rs.PVReadSparse(sys.argv[5], extended)
l3ow = rw.PVReadWeights(sys.argv[6])
l4oa = rs.PVReadSparse(sys.argv[7], extended)
l3fw = rw.PVReadWeights(sys.argv[8])
l4fa = rs.PVReadSparse(sys.argv[9], extended)


global l1wnxp
global l2wnxp
global l1weights
global l2weights
global lib2

l1anx = l1a.nx

"""
count = 0
count2 = 0

for h in range(10):
   for g in range(10): # nx - patch size + 1
      count3 = 0
      for i in range(4): # patch size
         for j in range(4): # patch size
            print "i + j = ", i * 10  + j + g + h * 10 # i * nx + j + g
            print "count3 = ", count3
            print "count2 = ", count2
            if (i + j) == 6:
               count2+=1
            count3+=1
            count+=1

         print
      print  "-----------"
print count
sys.exit()
"""

l1wnx = l1w.nx
l1wnxp = l1w.nxp
l2anx = l2a.nx
l2wnx = l2w.nx
l2wnxp = l2w.nxp
l3anx = l3a.nx
l3wnx = l3ow.nx
l3wnxp = l3ow.nxp
l4oanx = l4oa.nx
l4fanx = l4fa.nx

print
print "l4oanx = ", l4oanx
print "l3anx = ", l3anx
print "l2anx = ", l2anx
print "l2wnx = ", l2wnx
print "l2wnxp = ", l2wnxp
print "l1anx = ", l1anx
print "l1wnx = ", l1wnx
print "l1wnxp = ", l1wnxp
print






def nearby_neighbor(kzPre, zScaleLog2Pre, zScaleLog2Post):
   a = math.pow(2.0, (zScaleLog2Pre - zScaleLog2Post))
   ia = a

   if ia < 2:
      k0 = 0
   else:
      k0 = ia/2 - 1

   if a < 1.0 and kzPre < 0:
      k = kzPre - (1.0/a) + 1
   else:
      k = kzPre

   return k0 + (a * k)


def zPatchHead(kzPre, nzPatch, zScaleLog2Pre, zScaleLog2Post):
   a = math.pow(2.0, (zScaleLog2Pre - zScaleLog2Post))

   if a == 1:
      shift = -(0.5 * nzPatch)
      return shift + nearby_neighbor(kzPre, zScaleLog2Pre, zScaleLog2Post)

   shift = 1 - (0.5 * nzPatch)

   if (nzPatch % 2) == 0 and a < 1:
      kpos = (kzPre < 0)

      if kzPre < 0:
         kpos = -(1+kzPre)
      else:
         kpos = kzPre

      l = (2*a*kpos) % 2
      if kzPre < 0:
         shift -= l == 1
      else:
         shift -= l == 0
   elif (nzPatch % 2) == 1 and a < 1:
      shift = -(0.5 * nzPatch)

   neighbor = nearby_neighbor(kzPre, zScaleLog2Pre, zScaleLog2Post)

   if nzPatch == 1:
      return neighbor

   return shift + neighbor

global l1l
global l2l
global l3l
global l4l


l1l =  l1wnx / l4oanx
l2l =  l2wnx / l4oanx
l3l =  l3wnx / l4oanx
l4l =  l4oanx / l4oanx


print "l1l = ", l1l
print "l2l = ", l2l
print "l3l = ", l3l
print "l4l = ", l4l
print
print

for i in range(10):
   o = zPatchHead(i, l1wnxp, -math.log(l1l, 2), -math.log(l2l, 2))
   print "o = ", o

o = zPatchHead(52, l1wnxp, -math.log(l1l, 2), -math.log(l2l, 2))
print "o = ", o

print
print "fin"
sys.exit()
#############
"""
kxPreHead = zPatchHead(kxPost, nxPostPatch, post->getXScale(), pre->getXScale());
kyPreHead = zPatchHead(kyPost, nyPostPatch, post->getYScale(), pre->getYScale());
"""
#############



def l1(k, l2nx, l1nx, lib):
   #k = 0*l1nx + 0
   nx = k % l1nx
   ny = k / l1nx
   #print "l1wnxp = ", l1wnxp


   kxph = zPatchHead(nx, l1wnxp, -math.log(l1l, 2), -math.log(l2l, 2))
   kyph = zPatchHead(ny, l1wnxp, -math.log(l1l, 2), -math.log(l2l, 2))



   diff = l1nx / float(l2nx)

   #print
   #print "diff = ", diff


   cnx = (nx+9) / diff
   cny = (ny+9) / diff
   hrange = (l1wnxp / 2)

   #print "nx ny = ", nx, ny, cnx, cny

   #print "hrange = ", hrange

   patch = l1weights[k]
   patch = np.reshape(patch, (l1wnxp, l1wnxp))
   
   print "patch = ", patch

   ea = np.zeros((20, l1wnxp * l1wnxp))

   count = 0

   for i in range(20):
      i+=1

      pastsec = lib[-i, 0:]
      pastsec = np.reshape(pastsec, (l2nx, l2nx))

      test = pastsec[kyph:kyph+l1wnxp, kxph:kxph+l1wnxp]

      """
      print "k = ", k
      pastsec = np.random.random_integers(0, 100, (l2nx, l2nx))
      print "range", ny, ny+l1wnxp, nx, nx+l1wnxp
      test = pastsec[ny:ny+l1wnxp, nx:nx+l1wnxp]

      fig = plt.figure()

      ax = fig.add_subplot(111)

      ax.imshow(pastsec, cmap=cm.jet, interpolation='nearest', vmin=np.min(pastsec), vmax=np.max(pastsec))

      fig2 = plt.figure()
      ax2 = fig2.add_subplot(111)

      ax2.imshow(test, cmap=cm.jet, interpolation='nearest', vmin=np.min(pastsec), vmax=np.max(pastsec))


      plt.show()
      sys.exit()
      """

      print "test = ", np.shape(test)
      print test
      #print "test[]", test
      print "pastsec", np.shape(pastsec)
      print "ny, nx", ny, nx

      twee = 0
      if np.shape(test[0]) or np.shape(test[1]) != 9:
         twee +=1

      for h in range(np.shape(test)[0]):
         for j in range(np.shape(test)[1]):
            if test[h, j] > 0:
               count+=1
               w = patch[h, j]
               re = math.exp(-((i-1)/20.))
               re = w * re

               difh = np.shape(test[0]) - l1wnxp
               difj = np.shape(test[1]) - l1wnxp
               if twee != 0:
                  1
                  #print "difh = ", difh 
                  #print "difj = ", difj 

               ea[i-1, l1wnxp * (h+difh) + (j+difj)] = re
            if twee != 0:
               1
               #print "l1wnxp * h + j =", (l1wnxp * h + j)



   ea = np.sum(ea, axis = 0)

   print "ea = ", ea

   if np.sum(ea) > 0.0:
      ea = ea / float(np.sum(ea))

   if math.isnan(ea[0]) == True:
      print
      print "isnan == True"
      sys.exit()
   sys.exit()

   return ea



diff = l1anx / l2anx

end = 50

cliff = (l1anx / 2) * l1anx + ((l1anx / 2)-10)

lib = np.zeros((1, l2anx * l2anx))
lib2 = np.zeros((1, l3anx * l3anx))
lib3o = np.zeros((1, l4oanx * l4oanx))
lib3f = np.zeros((1, l4fanx * l4fanx))


"""
for i in range(prewnx*prewnx):
   if i == 0:
      preweights = prew.next_patch()
   else:
      a = prew.next_patch()
      preweights = np.vstack((preweights, a))

print preweights
print np.shape(preweights)
"""

for i in range(l1wnx*l1wnx):
   if i == 0:
      l1weights = l1w.next_patch()
   else:
      a = l1w.next_patch()
      l1weights = np.vstack((l1weights, a))
"""
for i in range(l2wnx*l2wnx):
   if i == 0:
      l2weights = l2w.next_patch()
   else:
      a = l2w.next_patch()
      l2weights = np.vstack((l2weights, a))


for i in range(l3wnx*l3wnx):
   if i == 0:
      l3oweights = l3ow.next_patch()
   else:
      a = l3ow.next_patch()
      l3oweights = np.vstack((l3oweights, a))
for i in range(l3wnx*l3wnx):
   if i == 0:
      l3fweights = l3fw.next_patch()
   else:
      a = l3fw.next_patch()
      l3fweights = np.vstack((l3fweights, a))
"""



print "cliff = ", cliff

l1ar = lm(((l1anx * l1anx), (l1wnxp * l1wnxp +1))) 


for i in range(end):
   l1A = l1a.next_record()
   l2A = l2a.next_activity()
   #l3A = l3a.next_activity()
   #l4oA = l4oa.next_activity()
   #l4fA = l4fa.next_activity()

   l2A = np.reshape(l2A, (1, l2anx*l2anx))
   lib = np.vstack((lib, l2A))

   #l3A = np.reshape(l3A, (1, l3anx*l3anx))
   #lib2 = np.vstack((lib2, l3A))

   #l4oA = np.reshape(l4oA, (1, l4oanx*l4oanx))
   #lib3o = np.vstack((lib3o, l4oA))

   #l4fA = np.reshape(l4fA, (1, l4fanx*l4fanx))
   #lib3f = np.vstack((lib3f, l4fA))

   if len(l1A) > 0:

      print "l1A = ", l1A
      for g in range(len(l1A)):
         a = l1(l1A[g], l2anx, l1anx, lib)

         #print "a = ", a

         #print "len(a) + 1 = ", len(a)+1
         for h in range(len(a)+1):
            #print "h",h
            if h == (len(a)):
               #print " h 1 = ", h
               l1ar[l1A[g],(h)] += 1
            else:
               #print "h = ", h
               if a[h] > 0.0:
                  l1ar[l1A[g], h] += a[h]
         #print "shape = ", np.shape(a)
         #print 

         #e = lm.toarray(l1ar)
         #print l1ar
         #print " l1A ", l1A
         #print e[l1A[g], 0:]
         #print np.shape(e[l1A[g], 0:])

         #print "g = ", g

      #e = lm.toarray(l1ar)
      #print "l1ar = ", l1ar
      #for l in range(len(l1A)):
      #   print e[l1A[l], 0:]
      #sys.exit()


e = lm.toarray(l1ar)


print "l1ar s = s =", np.shape(l1ar)
sys.exit()




#######################
#def con(e):
#######################

for i in range(l1anx * l1anx):
   if math.isnan(e[i, 0]) == True:
      print "e = ", e
      print "e[i] = ", e[i, 0:]
      print "i=", i
      print np.shape(e)
      print
      print "1st nan == True"
      sys.exit()




for i in range(l1anx * l1anx):
   if np.sum(e[i, 0:]) >= 1.0:
      print "e[i, 0:]", e[i, 0:]
      print "e[i,  l2wnxp*l2wnxp]",  e[i, l1wnxp*l1wnxp]
      e[i, 0:] = e[i, 0:] / e[i, l1wnxp * l1wnxp]
      

   if math.isnan(e[i, 0]) == True:
      print "e = ", e
      print "e[i] = ", e[i, 0:]
      print "i=", i
      print np.shape(e)
      print
      print "nan == True"
      sys.exit()




e = np.delete(e, l1wnxp * l1wnxp, axis=1)


gq = lm(((l1anx * l1anx), (l1anx * l1anx * l1wnxp * l1wnxp))) 

print "shape gq ", np.shape(gq)

count2 = 0
count = 0


for h in range(l1anx):
   for g in range(l1anx): # nx - patch size + 1
      count3 = 0
      for i in range(l1wnxp): # patch size
         for j in range(l1wnxp): # patch size
            where = i * l1anx + j + g + h * l1anx
            if count2 > ((l1anx*l1anx) - 2000):
               print "where = ", where
               print "count2 = ", count2
               print "count3 = ", count3
               print
            gq[count2, where] = e[count2, count3]
            #print "i + j = ", i * 10  + j + g + h * 10
            #print "count = ", count
            #print "count2 = ", count2
            if (i + j) == l1wnxp + l1wnxp - 2:
               count2+=1

            count+=1
            count3+=1

            #print
   #print  "-----------"
print
print "gq = ", gq
print np.shape(gq)
print "fin"

