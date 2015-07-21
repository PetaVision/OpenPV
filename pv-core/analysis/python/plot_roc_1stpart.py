
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


"""
Show how to modify the coordinate formatter to report the image "z"
value of the nearest pixel given x and y
"""
extended = False
vmax = 100.0 # Hz

if len(sys.argv) < 6:
   print "usage: plot_avg_activity filename1,[end_time step_time begin_time], On-weigh filename, Off-weight filename"
   sys.exit()

#if len(sys.argv) >= 6:
#   vmax = float(sys.argv[5])


a1 = rs.PVReadSparse(sys.argv[1], extended)
end = int(sys.argv[2])
step = int(sys.argv[3])
begin = int(sys.argv[4])
endtest = end
steptest = step
begintest = begin
#zetest = rs.PVReadSparse(sys.argv[21], extended)
w =  rw.PVReadWeights(sys.argv[5])
wO = rw.PVReadWeights(sys.argv[6])
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
margin = 15
marginend = w.nx - margin

pa = []


print "(begin, end, step, max) == ", begin, end, step, vmax


space = 1
nx  = w.nx
ny  = w.ny
nxp = w.nxp
nyp = w.nyp
nf = w.nf
d = np.zeros((4,4))
coord = 1
nx_im = nx * (nxp + space) + space
ny_im = ny * (nyp + space) + space
numpat = w.numPatches


im = np.zeros((nx_im, ny_im))
im[:,:] = (w.max - w.min) / 2.
countnum = 0

im2 = np.zeros((nx_im, ny_im))
im2[:,:] = (w.max - w.min) / 2.

thecount=0
A1pos = np.array([0,0])
minpos = np.array([0,0])
minlist = np.array([0,0])
countpos = 0

begintest = 0
steptest = 10
endtest = 10


 
for i in range(nx):
   for j in range(ny):
      if i > (margin-1) and i < (marginend):
         if j > (margin-1) and j < (marginend):
            #print "i = ", i
            #print "j = ", j
            if countpos == 0:
               A1pos = [i, j]
            else:
               A1pos = np.vstack((A1pos, [i, j]))
            countpos+=1





#a2.rewind()
co = 0
for g in range(2):
   if g == 0:
      countg = 0
      testgraph = []
      test = []
      numofsteps = 250
      #print A1pos
      #print np.shape(A1pos)
      #A1pos = np.vstack((A1pos, [0, 0]))
      for k in range(zerange):    ####### range(step)
         if k%1000 == 0:
            print "at ", k


         countg += 1
         A1A = a1.next_record()
         A1t = np.zeros((nx*ny))
         
         #print " k = ", k

         for h in range(len(A1A)):
            if (margin-1) < A1A[h] % nx < marginend:
               if (margin-1) < A1A[h] / nx < marginend:
                  A1t[A1A[h]]+= 1

         #if np.sum(test) > 0:
         #   print "test = ", test
         #   print "sum = ", sum(test)
         #print "A1t = ", A1t
         #print "mint = ", mint

         A1t = np.reshape(A1t, (nx, ny))
         lenm = len(A1t)
         
         for i in range(margin):
            dele = lenm - i - 1
            A1t = np.delete(A1t, dele, 0)
         for i in range(margin):
            dele = lenm - i - 1
            A1t = np.delete(A1t, dele, 1)
         for i in range(margin):
            dele = i
            A1t = np.delete(A1t, 0, 0)
         for i in range(margin):
            dele = i
            A1t = np.delete(A1t, 0, 1)

         nl = len(A1t) * len(A1t)
         A1t = np.reshape(A1t, (1, nl))

         d = k / numofsteps
         #print
         #print "A1t = ", A1t
         #print np.shape(A1t)

         if k >= (numofsteps*d) and k < ((numofsteps * d) + numofsteps):
            if k == (numofsteps * d):
               A1p = A1t
               #minp = mint
               thecount+=1
            else:
               A1p = np.add(A1p,A1t)
               #minp = np.vstack((minp, mint))
               #print "A1p = ", A1p
               #print "minp = ", minp
               #print
               thecount+=1
         if k == (numofsteps-1):
            A1q = A1p 
            #minq = minp.sum(axis=0)
         if k == ((numofsteps*d) + (numofsteps-1)) and k != (numofsteps-1):
            A1q = np.vstack((A1q, A1p))
            #minq = np.vstack((minq, minp.sum(axis=0)))
            #print "A1q = ", A1q
            #print "minq = ", minq



      print "a1q = ", A1q
      #print "minq = ", minq



      sh = np.shape(A1q)
      #minsh = np.shape(minq)
      print "shape = ", sh
      #print "minshape = ", minsh




      for i in range(sh[0]):
         z = i%4
         if i == 0:
            a = np.array([1])
            A1qf = np.array(A1q[i])
         if i != 0 and (z==0):
            a = np.vstack((a, 1))
            A1qf = np.vstack((A1qf, A1q[i]))
         if i != 0 and (z==1):
            a = np.vstack((a, 1))
            A1qf = np.vstack((A1qf, A1q[i]))

         #if z==4 or z==5 or z==6 or z==7 and i!= 0:
         #   a = np.vstack((a,0))
      #print "A1q shape = ", np.shape(A1q)
      #print "a shape = ", np.shape(a)

      #print A1q

      #print minqf


      #res = np.sum(A1q, axis=1)
      #minres = np.sum(minq, axis=1)
      #hist1 = np.zeros((np.max(res)/sh[1])+3, dtype=int)
      #hist2 = np.zeros((np.max(minres)/minsh[1])+3, dtype=int)

      print "A1q = ", A1qf
      print "A1q shape = ", np.shape(A1qf)


      np.savetxt("roc-info-%s.txt" %(sys.argv[1]), A1qf, fmt='%d', delimiter = ';')        

      sys.exit()
      fig = plt.figure()
      ax = fig.add_subplot(111)
      
      ax.plot(np.arange(len(hist1)), hist1, '-o', color='b')
      #ax.plot(np.arange(len(hist2)), hist2, '--o', color='b')

      #ax.plot(np.arange(len(hist0)), hist0, 'o', color='y')

      ax.set_xlabel('CLIQUE BINS')
      ax.set_ylabel('COUNT')
      ax.set_title('Clique Histogram')
      ax.set_xlim(0, 1+(np.max(res)/sh[1]))
      ax.grid(True)





      plt.show()



      sys.exit()
##################################################################

