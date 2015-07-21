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

if len(sys.argv) < 5:
   print "usage: plot_avg_activity filename1[end_time step_time begin_time], On-weigh filename, Off-weight filename"
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
w =  rw.PVReadWeights(sys.argv[5])
#wO = rw.PVReadWeights(sys.argv[6])
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
margin = 32
margin = 61

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
countpos = 0

#a2.rewind()
co = 0
for g in range(2):
   if g == 0:
      for end in range(begin+step, step+1, step):
         A1 = a1.avg_activity(begin, end)
         #AF = np.zeros((lenofo, lenofo))
         countpos = 0
         lenofo = len(A1)
         lenofb = lenofo * lenofo
         beingplotted = []
         for i in range(lenofo):
            for j in range(lenofo):
               #print A1[i, j]
               if i > margin and j > margin and i < nx - margin and j < ny - margin:
                  ij = (i * nx) + j
                  if countpos == 0:
                     A1pos = [i, j]
                  else:
                     A1pos = np.vstack((A1pos, [i, j]))
                  countpos+=1

      print "pos shape = ", np.shape(A1pos)
      print "A1pos = ", A1pos


      a1.rewind()


      countg = 0
      testgraph = []
      test = []
      numofsteps = 2000
      #print A1pos
      #print np.shape(A1pos)
      #A1pos = np.vstack((A1pos, [0, 0]))


      for k in range(zerange):    ####### range(step)
         if k%1000 == 0:
            print "at ", k
         A1t = []


         countg += 1
         A1A = a1.next_record()
         #A2A = a2.next_record()
         #A3A = a3.next_record()
         #A4A = a4.next_record()
         #A5A = a5.next_record()
         #A6A = a6.next_record()
         #A7A = a7.next_record()
         #A8A = a8.next_record()
         #A9A = a9.next_record()
         #A10A = a10.next_record()
         #A11A = a11.next_record()
         #A12A = a12.next_record()
         #A13A = a13.next_record()
         #A14A = a14.next_record()
         #A15A = a15.next_record()
         #A16A = a16.next_record()
         A1t = np.zeros((1, np.shape(A1pos)[0]))

#####
         for g in range(np.shape(A1pos)[0]):
            w = A1pos[g]
            i = w[0]
            j = w[1]
            for h in range(len(A1A)):
               if A1A[h] == ((lenofo * i) + j):
                  A1t[0, g] += 1


         #if np.sum(test) > 0:
         #   print "test = ", test
         #   print "sum = ", sum(test)
         #print "A1t = ", A1t
         d = k / numofsteps
         #print
         #print "A1t = ", A1t
         #print np.shape(A1t)

         if k >= (numofsteps*d) and k < ((numofsteps * d) + numofsteps):
            if k == (numofsteps * d):
               A1p = A1t
               thecount+=1
            else:
               A1p = np.vstack((A1p,A1t))
               thecount+=1
         if k == (numofsteps-1):
            A1q = A1p.sum(axis=0) 
         if k == ((numofsteps*d) + (numofsteps-1)) and k != (numofsteps-1):
            A1q = np.vstack((A1q, A1p.sum(axis=0)))

      sh = np.shape(A1q)
      print "shape = ", sh



      for i in range(sh[0]):
         z = i%2
         if i == 0:
            a = np.array([1])
         if i != 0 and (z==0):
            a = np.vstack((a, 1))
         if z==1 and i!= 0:
            a = np.vstack((a,0))
      #print "A1q shape = ", np.shape(A1q)
      #print "a shape = ", np.shape(a)


      res = np.sum(A1q, axis=1)
      hist1 = np.zeros((np.max(res)/sh[1])+3, dtype=int)
      hist2 = np.zeros((np.max(res)/sh[1])+3, dtype=int)

      for i in range(len(res)):
         z = i%2
         if z==0:
            ph = ((res[i])/float(sh[1]))
            hist1[ph] += 1
         if z==1:
            ph = (res[i]/float(sh[1]))
            hist2[ph] += 1

      A1q = np.insert(A1q, [0], a, axis=1)

      np.savetxt("roc-info.txt", A1q, fmt='%d', delimiter = ';')        


      fig = plt.figure()
      ax = fig.add_subplot(111)
      
      ax.plot(np.arange(len(hist1)), hist1, '-o', color='b')
      ax.plot(np.arange(len(hist2)), hist2, '--o', color='b')

      #ax.plot(np.arange(len(hist0)), hist0, 'o', color='y')

      ax.set_xlabel('CLIQUE BINS')
      ax.set_ylabel('COUNT')
      ax.set_title('Clique Histogram')
      ax.set_xlim(0, 1+(np.max(res)/sh[1]))
      ax.grid(True)





      plt.show()



      sys.exit()
##################################################################
