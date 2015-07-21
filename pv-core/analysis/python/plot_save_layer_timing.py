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




def l3o(k, l3nx):
   print "at l3o"
   k = int(k)
   nx = k % l3nx
   ny = k / l3nx

   l4nx = l4oa.nx

   diff = l3nx / float(l4nx)

   cnx = nx / diff
   cny = ny / diff
   hrange = (l3wnxp / 2)   

   patch = l3oweights[k]
   patch = np.reshape(patch, (l3wnxp, l3wnxp))

   count = 0
   for i in range(20):
      i+=1

      pastsec = lib3o[-i, 0:]
      pastsec = np.reshape(pastsec, (l4nx, l4nx))
      
      test = pastsec[cny-hrange:cny+hrange+1, cnx-hrange:cnx+hrange+1]

      print
      print "cny = ", cny
      print "cnx = ", cnx
      print "hrange = ", hrange
      print "diff = ", diff
      print "nx = ", nx
      print "ny = ", ny
      print "shape of patch = ", np.shape(patch)
      print "shape 1 = ", np.shape(test)[0]
      print "shape 2 = ", np.shape(test)[1]
      print "l3nx = ", l3nx
      print "l4nx = ", l4nx
      print


      for h in range(len(test)):
         for j in range(len(test)):
            if test[h, j] > 0:
               count+=1
               w = patch[h, j]
               re = math.exp(-((i-1)/20.))
               re = w * re
               if count == 1:
                  fin = re
               else:
                  fin = fin * re
   return fin


def l3f(k, l3nx):
   print "at l3f"
   k = int(k)
   nx = k % l3nx
   ny = k / l3nx

   l4nx = l4fa.nx

   diff = l3nx / float(l4nx)

   cnx = nx / diff
   cny = ny / diff
   hrange = (l3wnxp / 2)   

   patch = l3fweights[k]
   patch = np.reshape(patch, (l3wnxp, l3wnxp))

   count = 0
   for i in range(20):
      i+=1

      pastsec = lib3f[-i, 0:]
      pastsec = np.reshape(pastsec, (l4nx, l4nx))
      
      test = pastsec[cny-hrange:cny+hrange+1, cnx-hrange:cnx+hrange+1]

      for h in range(len(test)):
         for j in range(len(test)):
            if test[h, j] > 0:
               count+=1
               w = patch[h, j]
               re = math.exp(-((i-1)/20.))
               re = w * re
               if count == 1:
                  fin = re
               else:
                  fin = fin * re
   return fin





def l2(k, l2nx):
   print "at l2"
   k = int(k)
   nx = k % l2nx
   ny = k / l2nx

   l3nx = l3a.nx

   diff = l2nx / float(l3nx)

   cnx = nx / diff
   cny = ny / diff
   hrange = (l2wnxp / 2)   

   patch = l2weights[k]
   patch = np.reshape(patch, (l2wnxp, l2wnxp))

   ea = np.zeros((20, l2wnxp * l2wnxp))

   count = 0
   for i in range(20):
      i+=1
      #print "i = ", i
      pastsec = lib2[-i, 0:]
      pastsec = np.reshape(pastsec, (l3nx, l3nx))

      test = pastsec[cny-hrange:cny+hrange, cnx-hrange:cnx+hrange]

      for h in range(len(test)):
         for j in range(len(test)):

            if test[h, j] > 0:
               count+=1

               w = patch[h, j]
               re = math.exp(-((i-1)/20.))
               re = w * re

               ea[i-1, l2wnxp * h + j] = re

               #wherey = cny - hrange + h
               #wherex = cnx - hrange + j
               #newk = wherey * l2anx + wherex
               #reso = l3o(newk, l3nx)
               #resf = l3f(newk, l3nx)
               #if count == 1:
               #   res = reso * resf * re
               #else:
               #   res = res * reso * resf * re
   #return res
   ea = np.sum(ea, axis = 0)
   ea = ea / float(np.sum(ea))

   return ea



def l1(k, l2nx, l1nx, lib):
   #k = 0*l1nx + 0
   nx = k % l1nx
   ny = k / l1nx
   #print "l1wnxp = ", l1wnxp



   diff = l1nx / float(l2nx)

   print "diff = ", diff
   sys.exit()

   #print
   #print "diff = ", diff


   cnx = (nx+9) / diff
   cny = (ny+9) / diff
   hrange = (l1wnxp / 2)

   #print "nx ny = ", nx, ny, cnx, cny

   #print "hrange = ", hrange

   patch = l1weights[k]
   patch = np.reshape(patch, (l1wnxp, l1wnxp))
   


   ea = np.zeros((20, l1wnxp * l1wnxp))

   count = 0

   for i in range(20):
      i+=1

      pastsec = lib[-i, 0:]
      pastsec = np.reshape(pastsec, (l2nx, l2nx))

      test = pastsec[ny:ny+l1wnxp, nx:nx+l1wnxp]

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

   if np.sum(ea) > 0.0:
      ea = ea / float(np.sum(ea))

   if math.isnan(ea[0]) == True:
      print
      print "isnan == True"
      sys.exit()


   return ea




print "l2anx = ", l2anx
print "l2wnx = ", l2wnx
print "l2wnxp = ", l2wnxp
print "l1anx = ", l1anx
print "l1wnx = ", l1wnx
print "l1wnxp = ", l1wnxp
print

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

