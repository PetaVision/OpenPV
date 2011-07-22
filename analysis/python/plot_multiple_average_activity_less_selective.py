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

if len(sys.argv) < 22:
   print "usage: plot_avg_activity filename1, filename2, filename3, filename4, filename5, filename6, filename7, filename8, filename9, filename10, filename11, filename12, filename13, filename14, filename15, filename16 [end_time step_time begin_time], test filename, On-weigh filename, Off-weight filename"
   sys.exit()

#if len(sys.argv) >= 6:
#   vmax = float(sys.argv[5])


a1 = rs.PVReadSparse(sys.argv[1], extended)
a2 = rs.PVReadSparse(sys.argv[2], extended)
a3 = rs.PVReadSparse(sys.argv[3], extended)
a4 = rs.PVReadSparse(sys.argv[4], extended)
a5 = rs.PVReadSparse(sys.argv[5], extended)
a6 = rs.PVReadSparse(sys.argv[6], extended)
a7 = rs.PVReadSparse(sys.argv[7], extended)
a8 = rs.PVReadSparse(sys.argv[8], extended)
a9 = rs.PVReadSparse(sys.argv[9], extended)
a10 = rs.PVReadSparse(sys.argv[10], extended)
a11 = rs.PVReadSparse(sys.argv[11], extended)
a12 = rs.PVReadSparse(sys.argv[12], extended)
a13 = rs.PVReadSparse(sys.argv[13], extended)
a14 = rs.PVReadSparse(sys.argv[14], extended)
a15 = rs.PVReadSparse(sys.argv[15], extended)
a16 = rs.PVReadSparse(sys.argv[16], extended)
end = int(sys.argv[17])
step = int(sys.argv[18])
begin = int(sys.argv[19])
endtest = end
steptest = step
begintest = begin
atest = rs.PVReadSparse(sys.argv[20], extended)
w = rw.PVReadWeights(sys.argv[21])
wO = rw.PVReadWeights(sys.argv[22])
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

A1pos = np.array([0,0])


pa = []

print "(begin, end, step, max) == ", begin, end, step, vmax


for endtest in range(begintest+steptest, steptest+1, steptest):
   Atest = atest.avg_activity(begintest, endtest)
   lenofo = len(Atest)
   for i in range(lenofo):
      for j in range(lenofo):
         pa = np.append(pa, Atest[i,j])  
median = np.median(pa)
avg = np.mean(pa)

AW = np.zeros((lenofo, lenofo))
AWO = np.zeros((lenofo, lenofo))
SUMAW = np.zeros((lenofo, lenofo))
whereA1 = np.zeros((lenofo, lenofo))


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
margin = 20

im = np.zeros((nx_im, ny_im))
im[:,:] = (w.max - w.min) / 2.


im2 = np.zeros((nx_im, ny_im))
im2[:,:] = (w.max - w.min) / 2.


print "avg = ", avg
print "median = ", median
#a2.rewind()
co = 0
for g in range(2):
   if g == 0:
      for end in range(begin+step, step+1, step):
         countpos = 0
         countposcomp = 0
         A1 = a1.avg_activity(begin, end)
         A2 = a2.avg_activity(begin, end)
         A3 = a3.avg_activity(begin, end)
         A4 = a4.avg_activity(begin, end)
         A5 = a5.avg_activity(begin, end)
         A6 = a6.avg_activity(begin, end)
         A7 = a7.avg_activity(begin, end)
         A8 = a8.avg_activity(begin, end)
         A9 = a9.avg_activity(begin, end)
         A10 = a10.avg_activity(begin, end)
         A11 = a11.avg_activity(begin, end)
         A12 = a12.avg_activity(begin, end)
         A13 = a13.avg_activity(begin, end)
         A14 = a14.avg_activity(begin, end)
         A15 = a15.avg_activity(begin, end)
         A16 = a16.avg_activity(begin, end)
         AF = np.zeros((lenofo, lenofo))

         lenofo = len(A1)
         lenofb = lenofo * lenofo
         beingplotted = []
         for i in range(lenofo):
            for j in range(lenofo):
               #print A1[i, j]
               check = [A1[i,j], A2[i,j], A3[i,j], A4[i,j], A5[i,j], A6[i,j], A7[i,j], A8[i,j], A9[i,j], A10[i,j], A11[i,j], A12[i,j], A13[i,j], A14[i,j], A15[i,j], A16[i,j]]

               checkmax = np.max(check)
               wheremax = np.argmax(check)
               half = checkmax / 2.0
               sort = np.sort(check)
               co = 0
               if wheremax == 0:
                  AW[i, j] = 1
               if wheremax == 1:
                  AW[i, j] = 2
               if wheremax == 2:
                  AW[i, j] = 3
               if wheremax == 3:
                  AW[i, j] = 4
               if wheremax == 4:
                  AW[i, j] = 5
               if wheremax == 5:
                  AW[i, j] = 6
               if wheremax == 6:
                  AW[i, j] = 7
               if wheremax == 7:
                  AW[i, j] = 8
               if wheremax == 8:
                  AW[i, j] = 9
               if wheremax == 9:
                  AW[i, j] = 10
               if wheremax == 10:
                  AW[i, j] = 11
               if wheremax == 11:
                  AW[i, j] = 12
               if wheremax == 12:
                  AW[i, j] = 13
               if wheremax == 13:
                  AW[i, j] = 14
               if wheremax == 14:
                  AW[i, j] = 15
               if wheremax == 15:
                  AW[i, j] = 16
               

               #print AF[i, j]
               #print "check = ", sort
               #print "half = ", half
               for e in range(len(check)):
                  if check[e] >= half:
                     co += 1
               if co == 1:
                  AF[i, j] = 0.0
                  count1 += 1
                  AWO[i, j] = 1.0
                  if wheremax == 0:
                     if i > margin and i < (w.nx - margin):
                        if j > margin and j < (w.ny - margin):
                           if countpos == 0:
                              A1pos = [i, j]
                           else:
                              A1pos = np.vstack((A1pos, [i, j]))
                           countpos+=1   

          
               elif co == 2:
                  AF[i, j] = 0.06
                  count2 += 1
                  AWO[i, j] = 2.0
                  if wheremax == 0:
                     if i > margin and i < (w.nx - margin):
                        if j > margin and j < (w.ny - margin):
                           if countposcomp == 0:
                              A1comp = [i, j]
                           else:
                              A1comp = np.vstack((A1comp, [i, j]))
                           countposcomp+=1   



               elif co == 3:
                  AF[i, j] = 0.12
                  count3 += 1
                  AWO[i, j] = 3.0


               elif co == 4:
                  AF[i, j] = 0.18
                  count4 += 1
                  AWO[i, j] = 4.0


               elif co == 5:
                  AF[i, j] = 0.24
                  count5 += 1
                  AWO[i, j] = 5.0



               elif co == 6:
                  AF[i, j] = 0.3
                  count6 += 1
                  AWO[i, j] = 6.0
#######
                  #if A1[i ,f]
#######


               elif co == 7:
                  AF[i, j] = 0.36
                  count7 += 1
                  AWO[i, j] = 7.0


               elif co == 8:
                  AF[i, j] = 0.42
                  count8 += 1
                  AWO[i, j] = 8.0


               elif co == 9:
                  AF[i, j] = 0.48
                  count9 += 1
                  AWO[i, j] = 9.0


               elif co == 10:
                  AF[i, j] = 0.54
                  count10 += 1
                  AWO[i, j] = 10.0


               elif co == 11:
                  AF[i, j] = 0.60
                  count11 += 1
                  AWO[i, j] = 11.0


               elif co == 12:
                  AF[i, j] = 0.66
                  count12 += 1
                  AWO[i, j] = 12.0


               elif co == 13:
                  AF[i, j] = 0.72
                  count13 += 1
                  AWO[i, j] = 13.0

               elif co == 14:
                  AF[i, j] = 0.78
                  count14 += 1
                  AWO[i, j] = 14.0
               elif co == 15:
                  AF[i, j] = 0.84
                  count15 += 1
                  AWO[i, j] = 15.0
               elif co == 16:
                  AF[i, j] = 0.9
                  count16 += 1
                  AWO[i, j] = 16.0
               else:
                  AF[i, j] = 1.0
                  count18 += 1
                  #print "ELSE"
               #print "co = ", co
               #print
               #print AF[i ,j]
               #print 
         #print "13", count13
         #print "14", count14
         #print "15", count15
         #print "16", count16

      a1.rewind()
      a2.rewind()
      a3.rewind()
      a4.rewind()
      a5.rewind()
      a6.rewind()
      a7.rewind()
      a8.rewind()
      a9.rewind()
      a10.rewind()
      a11.rewind()
      a12.rewind()
      a13.rewind()
      a14.rewind()
      a15.rewind()
      a16.rewind()

      countg = 0
      testgraph = []
      test = []
      numofsteps = 2000
      for k in range(zerange):    ####### range(step)
         if k%1000 == 0:
            print "at ", k
         A1t = []
         A1lt = []
         A2t = []
         A3t = []
         A4t = []
         A5t = []
         A6t = []
         A7t = []
         A8t = []
         A9t = []
         A10t = []
         A11t = []
         A12t = []
         A13t = []
         A14t = []
         A15t = []
         A16t = []


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
         for g in range(np.shape(A1pos)[0]):
            w = A1pos[g]
            i = w[0]
            j = w[1]
            for h in range(len(A1A)):
               if A1A[h] == ((lenofo * i) + j):
                  A1t[0,g] += 1
         A1tcomp = np.zeros((1, np.shape(A1comp)[0]))
         count1 = 0
         for g in range(np.shape(A1comp)[0]):
            w = A1comp[g]
            i = w[0]
            j = w[1]
            for h in range(len(A1A)):
               if A1A[h] == ((lenofo * i) + j):
                  #print "count1 = ", count1
                  #print "a1tcomp = ", A1tcomp
                  #print "g = ", g
                  A1tcomp[0,g] += 1
                  count1 +=1

                  """
                  if AW[i, j] == 2:
                     t = 0
                     for h in range(len(A2A)):
                        if A2A[h] == ((lenofo * i) + j):
                           t = 1
                     if t ==1:
                        A2t = np.append(A2t,1)
                     else:
                        A2t = np.append(A2t, 0)
                  if AW[i, j] == 3:
                     t = 0
                     for h in range(len(A3A)):
                        if A3A[h] == ((lenofo * i) + j):
                           t = 1
                     if t == 1:
                        A3t = np.append(A3t,1)
                     else:
                        A3t = np.append(A3t, 0)
                  if AW[i, j] == 4:
                     t = 0
                     for h in range(len(A4A)):
                        if A4A[h] == ((lenofo * i) + j):
                           t = 1
                     if t == 1:
                        A4t = np.append(A4t,1)
                     else:
                        A4t = np.append(A4t, 0)
                  if AW[i, j] == 5:
                     t = 0
                     for h in range(len(A5A)):
                        if A5A[h] == ((lenofo * i) + j):
                           t = 1
                     if t == 1:
                        A5t = np.append(A5t,1)
                     else:
                        A5t = np.append(A5t, 0)
                  if AW[i, j] == 6:
                     t = 0
                     for h in range(len(A6A)):
                        if A6A[h] == ((lenofo * i) + j):
                           t = 1
                     if t == 1:
                        A6t = np.append(A6t,1)
                     else:
                        A6t = np.append(A6t, 0)
                  if AW[i, j] == 7:
                     t = 0
                     for h in range(len(A7A)):
                        if A7A[h] == ((lenofo * i) + j):
                           t = 1
                     if t == 1:
                        A7t = np.append(A7t,1)
                     else:
                        A7t = np.append(A7t, 0)
                  if AW[i, j] == 8:
                     t = 0
                     for h in range(len(A8A)):
                        if A8A[h] == ((lenofo * i) + j):
                           t = 1
                     if t == 1:
                        A8t = np.append(A8t,1)
                     else:
                        A8t = np.append(A8t, 0)
                  if AW[i, j] == 9:
                     t = 0
                     for h in range(len(A9A)):
                        if A9A[h] == ((lenofo * i) + j):
                           t = 1
                     if t == 1:
                        A9t = np.append(A9t,1)
                     else:
                        A9t = np.append(A9t, 0)
                  if AW[i, j] == 10:
                     t = 0
                     for h in range(len(A10A)):
                        if A10A[h] == ((lenofo * i) + j):
                           t = 1
                     if t == 1:
                        A10t = np.append(A10t,1)
                     else:
                        A10t = np.append(A10t, 0)
                  if AW[i, j] == 11:
                     t = 0
                     for h in range(len(A11A)):
                        if A11A[h] == ((lenofo * i) + j):
                           t = 1
                     if t == 1:
                        A11t = np.append(A11t,1)
                     else:
                        A11t = np.append(A11t, 0)
                  if AW[i, j] == 12:
                     t = 0
                     for h in range(len(A12A)):
                        if A12A[h] == ((lenofo * i) + j):
                           t = 1
                     if t == 1:
                        A12t = np.append(A12t,1)
                     else:
                        A12t = np.append(A12t, 0)
                  if AW[i, j] == 13:
                     t = 0
                     for h in range(len(A13A)):
                        if A13A[h] == ((lenofo * i) + j):
                           t = 1
                     if t == 1:
t                        A13t = np.append(A13t,1)
                     else:
                        A13t = np.append(A13t, 0)
                  if AW[i, j] == 14:
                     t = 0
                     for h in range(len(A14A)):
                        if A14A[h] == ((lenofo * i) + j):
                           t = 1
                     if t == 1:
                        A14t = np.append(A14t,1)
                     else:
                        A14t = np.append(A14t, 0)
                  if AW[i, j] == 15:
                     t = 0
                     for h in range(len(A15A)):
                        if A15A[h] == ((lenofo * i) + j):
                           t = 1
                     if t == 1:
3                        A15t = np.append(A15t,1)
                     else:
                        A15t = np.append(A15t, 0)
t                  if AW[i, j] == 16:
                     t = 0
                     for h in range(len(A16A)):
                        if A16A[h] == ((lenofo * i) + j):
                           t = 1
                     if t == 1:
                        A16t = np.append(A16t,1)
                     else:
                        A16t = np.append(A16t, 0)
                  """

         #if np.sum(test) > 0:
         #   print "test = ", test
         #   print "sum = ", sum(test)

         d = k / numofsteps


         if k > numofsteps and k != zerange - 1:
            if (k+1)%(numofsteps*2) == 0:
               if np.shape(A1t)[1] < np.shape(A1tcomp)[1]:
                  A1tcomp = np.split(A1tcomp[0], [np.shape(A1t)[1]-np.shape(A1tcomp)[1]])
                  A1p = A1t
                  A1pcomp = A1tcomp[0]
               elif np.shape(A1t)[1] > np.shape(A1tcomp)[1]:
                  A1t = np.split(A1t[0], [np.shape(A1tcomp)[1]-np.shape(A1t)[1]])
                  A1p = A1t[0]             
                  A1pcomp = A1tcomp
               else:
                  A1p = A1t
                  A1pcomp = A1tcomp

            if k%(numofsteps*2)+1 > 0 and k%(numofsteps*2)+1 <= numofsteps:
               if np.shape(A1t)[1] < np.shape(A1tcomp)[1]:
                  A1tcomp = np.split(A1tcomp[0], [np.shape(A1t)[1]-np.shape(A1tcomp)[1]])
                  A1p = np.vstack((A1p, A1t))
                  A1pcomp = np.vstack((A1pcomp, A1tcomp[0]))
               elif np.shape(A1t)[1] > np.shape(A1tcomp)[1]:
                  A1t = np.split(A1t[0], [np.shape(A1tcomp)[1]-np.shape(A1t)[1]])
                  A1p = np.vstack((A1p, A1t[0]))
                  A1pcomp = np.vstack((A1pcomp, A1tcomp))
                  print A1pcomp
               else:
                  A1p = np.vstack((A1p, A1t))
                  A1pcomp = np.vstack((A1pcomp, A1tcomp))
        

            if k == (numofsteps*3)-1:

               A1q = A1p.sum(axis=0)
               A1q = np.vstack((A1q, A1pcomp.sum(axis=0)))
            if k%(numofsteps*2)+1 == numofsteps and k != (d*3)-1:
               A1q = np.vstack((A1q, A1p.sum(axis=0)))
               A1q = np.vstack((A1q, A1pcomp.sum(axis=0)))




         """
##########
         if k >= (numofsteps*d) and k < ((numofsteps * d) + numofsteps):
            if k == (numofsteps * d):
               A2p = np.sum(A2t)
            else:
               A2p = np.append(A2p,np.sum(A2t))
         if k == (numofsteps-1):
            A2q = np.average(A2p) 
         if k == ((numofsteps*d) + (numofsteps-1)) and k != (numofsteps-1):
            A2q = np.append(A2q, np.average(A2p))

##########
         if k >= (numofsteps*d) and k < ((numofsteps * d) + numofsteps):
            if k == (numofsteps * d):
               A3p = np.sum(A3t)
            else:
               A3p = np.append(A3p,np.sum(A3t))
         if k == (numofsteps-1):
            A3q = np.average(A3p) 
         if k == ((numofsteps*d) + (numofsteps-1)) and k != (numofsteps-1):
            A3q = np.append(A3q, np.average(A3p))

##########
         if k >= (numofsteps*d) and k < ((numofsteps * d) + numofsteps):
            if k == (numofsteps * d):
               A4p = np.sum(A4t)
            else:
               A4p = np.append(A4p,np.sum(A4t))
         if k == (numofsteps-1):
            A4q = np.average(A4p) 
         if k == ((numofsteps*d) + (numofsteps-1)) and k != (numofsteps-1):
            A4q = np.append(A4q, np.average(A4p))

##########
         if k >= (numofsteps*d) and k < ((numofsteps * d) + numofsteps):
            if k == (numofsteps * d):
               A5p = np.sum(A5t)
            else:
               A5p = np.append(A5p,np.sum(A5t))
         if k == (numofsteps-1):
            A5q = np.average(A5p) 
         if k == ((numofsteps*d) + (numofsteps-1)) and k != (numofsteps-1):
            A5q = np.append(A5q, np.average(A5p))

##########
         if k >= (numofsteps*d) and k < ((numofsteps * d) + numofsteps):
            if k == (numofsteps * d):
               A6p = np.sum(A6t)
            else:
               A6p = np.append(A6p,np.sum(A6t))
         if k == (numofsteps-1):
            A6q = np.average(A6p) 
         if k == ((numofsteps*d) + (numofsteps-1)) and k != (numofsteps-1):
            A6q = np.append(A6q, np.average(A6p))

##########
         if k >= (numofsteps*d) and k < ((numofsteps * d) + numofsteps):
            if k == (numofsteps * d):
               A7p = np.sum(A7t)
            else:
               A7p = np.append(A7p,np.sum(A7t))
         if k == (numofsteps-1):
            A7q = np.average(A7p) 
         if k == ((numofsteps*d) + (numofsteps-1)) and k != (numofsteps-1):
            A7q = np.append(A7q, np.average(A7p))

##########
         if k >= (numofsteps*d) and k < ((numofsteps * d) + numofsteps):
            if k == (numofsteps * d):
               A8p = np.sum(A8t)
            else:
               A8p = np.append(A8p,np.sum(A8t))
         if k == (numofsteps-1):
            A8q = np.average(A8p) 
         if k == ((numofsteps*d) + (numofsteps-1)) and k != (numofsteps-1):
            A8q = np.append(A8q, np.average(A8p))

##########
         if k >= (numofsteps*d) and k < ((numofsteps * d) + numofsteps):
            if k == (numofsteps * d):
               A9p = np.sum(A9t)
            else:
               A9p = np.append(A9p,np.sum(A9t))
         if k == (numofsteps-1):
            A9q = np.average(A9p) 
         if k == ((numofsteps*d) + (numofsteps-1)) and k != (numofsteps-1):
            A9q = np.append(A9q, np.average(A9p))

##########
         if k >= (numofsteps*d) and k < ((numofsteps * d) + numofsteps):
            if k == (numofsteps * d):
               A10p = np.sum(A10t)
            else:
               A10p = np.append(A10p,np.sum(A10t))
         if k == (numofsteps-1):
            A10q = np.average(A10p) 
         if k == ((numofsteps*d) + (numofsteps-1)) and k != (numofsteps-1):
            A10q = np.append(A10q, np.average(A10p))

##########
         if k >= (numofsteps*d) and k < ((numofsteps * d) + numofsteps):
            if k == (numofsteps * d):
               A11p = np.sum(A11t)
            else:
               A11p = np.append(A11p,np.sum(A11t))
         if k == (numofsteps-1):
            A11q = np.average(A11p) 
         if k == ((numofsteps*d) + (numofsteps-1)) and k != (numofsteps-1):
            A11q = np.append(A11q, np.average(A11p))

##########
         if k >= (numofsteps*d) and k < ((numofsteps * d) + numofsteps):
            if k == (numofsteps * d):
               A12p = np.sum(A12t)
            else:
               A12p = np.append(A12p,np.sum(A12t))
         if k == (numofsteps-1):
            A12q = np.average(A12p) 
         if k == ((numofsteps*d) + (numofsteps-1)) and k != (numofsteps-1):
            A12q = np.append(A12q, np.average(A12p))

##########
         if k >= (numofsteps*d) and k < ((numofsteps * d) + numofsteps):
            if k == (numofsteps * d):
               A13p = np.sum(A13t)
            else:
               A13p = np.append(A13p,np.sum(A13t))
         if k == (numofsteps-1):
            A13q = np.average(A13p) 
         if k == ((numofsteps*d) + (numofsteps-1)) and k != (numofsteps-1):
            A13q = np.append(A13q, np.average(A13p))

##########
         if k >= (numofsteps*d) and k < ((numofsteps * d) + numofsteps):
            if k == (numofsteps * d):
               A14p = np.sum(A14t)
            else:
               A14p = np.append(A14p,np.sum(A14t))
         if k == (numofsteps-1):
            A14q = np.average(A14p) 
         if k == ((numofsteps*d) + (numofsteps-1)) and k != (numofsteps-1):
            A14q = np.append(A14q, np.average(A14p))

##########
         if k >= (numofsteps*d) and k < ((numofsteps * d) + numofsteps):
            if k == (numofsteps * d):
               A15p = np.sum(A15t)
            else:
               A15p = np.append(A15p,np.sum(A15t))
         if k == (numofsteps-1):
            A15q = np.average(A15p) 
         if k == ((numofsteps*d) + (numofsteps-1)) and k != (numofsteps-1):
            A15q = np.append(A15q, np.average(A15p))

##########
         if k >= (numofsteps*d) and k < ((numofsteps * d) + numofsteps):
            if k == (numofsteps * d):
               A16p = np.sum(A16t)
            else:
               A16p = np.append(A16p,np.sum(A16t))
         if k == (numofsteps-1):
            A16q = np.average(A16p) 
         if k == ((numofsteps*d) + (numofsteps-1)) and k != (numofsteps-1):
            A16q = np.append(A16q, np.average(A16p))

         """




         #for i in range(4):
         #   testq = np.append(testq, 0)
                     #if AW[i, j] == 2:
                     #   for g in range(len(A2A)):
                     #      if A2A[g] == ((4 * i) + j):
                     #         SUMAW[i, j] += 1
                     #if AW[i, j] == 3:
                     #   for g in range(len(A3A)):
                     #      if A3A[g] == ((4 * i) + j):
                     #         SUMAW[i, j] += 1
                     #if AW[i, j] == 4:
                     #   for g in range(len(A4A)):
                     #      if A4A[g] == ((4 * i) + j):
                     #         SUMAW[i, j] += 1
                     #if AW[i, j] == 5:
                     #   for g in range(len(A5A)):
                     #      if A5A[g] == ((4 * i) + j):
                     #         SUMAW[i, j] += 1
                     #if AW[i, j] == 6:
                     #   for g in range(len(A6A)):
                     #      if A6A[g] == ((4 * i) + j):
                     #         SUMAW[i, j] += 1
                     #if AW[i, j] == 7:
                     #   for g in range(len(A7A)):
                     #      if A7A[g] == ((4 * i) + j):
                     #         SUMAW[i, j] += 1
                     #if AW[i, j] == 8:
                     #   for g in range(len(A8A)):
                     #      if A8A[g] == ((4 * i) + j):
                     #         SUMAW[i, j] += 1
                     #if AW[i, j] == 9:
                     #   for g in range(len(A9A)):
                     #      if A9A[g] == ((4 * i) + j):
                     #         SUMAW[i, j] += 1
                     #if AW[i, j] == 10:
                     #   for g in range(len(A10A)):
                     #      if A10A[g] == ((4 * i) + j):
                     #         SUMAW[i, j] += 1
                     #if AW[i, j] == 11:
                     #   for g in range(len(A11A)):
                     #      if A11A[g] == ((4 * i) + j):
                     #         SUMAW[i, j] += 1
                     #if AW[i, j] == 12:
                     #   for g in range(len(A12A)):
                     #      if A12A[g] == ((4 * i) + j):
                     #         SUMAW[i, j] += 1
                     #if AW[i, j] == 13:
                     #   for g in range(len(A13A)):
                     #      if A13A[g] == ((4 * i) + j):
                     #         SUMAW[i, j] += 1
                     #if AW[i, j] == 14:
                     #   for g in range(len(A14A)):
                     #      if A14A[g] == ((4 * i) + j):
                     #         SUMAW[i, j] += 1
                     #f AW[i, j] == 15:
                     #   for g in range(len(A15A)):
                     #      if A15A[g] == ((4 * i) + j):
                     #         SUMAW[i, j] += 1
                     #if AW[i, j] == 16:
                     #   for g in range(len(A16A)):
                     #      if A16A[g] == ((4 * i) + j):
                     #         SUMAW[i, j] += 1


         #fig = plt.figure()
         #ax = fig.add_subplot(111)
         #ax.set_title("SUMAW")
         #ax.imshow(SUMAW, cmap=cm.binary, interpolation='nearest')

         #test = SUMAW / countg



      #A1q = (A1q / len(A1t)) / (numofsteps / 1000.0)
      #A1lq = (A1lq / len(A1lt)) / (numofsteps / 1000.0)
      #A2q = (A2q / len(A2t)) / (numofsteps / 1000.0)
      #A3q = (A3q / len(A3t)) / (numofsteps / 1000.0)
      #A4q = (A4q / len(A4t)) / (numofsteps / 1000.0)
      #A5q = (A5q / len(A5t)) / (numofsteps / 1000.0)
      #A6q = (A6q / len(A6t)) / (numofsteps / 1000.0)
      #A7q = (A7q / len(A7t)) / (numofsteps / 1000.0)
      #A8q = (A8q / len(A8t)) / (numofsteps / 1000.0)
      #A9q = (A9q / len(A9t)) / (numofsteps / 1000.0)
      #A10q = (A10q / len(A10t)) / (numofsteps / 1000.0)
      #A11q = (A11q / len(A11t)) / (numofsteps / 1000.0)
      #A12q = (A12q / len(A12t)) / (numofsteps / 1000.0)
      #A13q = (A13q / len(A13t)) / (numofsteps / 1000.0)
      #A14q = (A14q / len(A14t)) / (numofsteps / 1000.0)
      #A15q = (A15q / len(A15t)) / (numofsteps / 1000.0)
      #A16q = (A16q / len(A16t)) / (numofsteps / 1000.0)


      #f = open('averaged-activity.txt', 'w')
      #for l in range(2000): #(len(A1q)/2)):
      #   f.write('1; %1.1f\n' %(A1q[l]))
      #for l in range(2000): #((len(A1q)/2)):
      #   f.write('0; %1.1f\n' %(A1lq[l]))

      sh = np.shape(A1q)
      print "shape = ", sh

      for i in range(sh[0]):
         if i == 0:
            a = np.array([1])
         if i%2 == 0 and i != 0:
            a = np.vstack((a, 1))
         if i%2 != 0 and i != 0:
            a = np.vstack((a, 0))
      A1q = np.insert(A1q, [0], a, axis=1)


      np.savetxt("roc-info.txt", A1q, fmt='%d', delimiter = ';')  
      print "A1q = ", A1q


      res = np.sum(A1q, axis=1)
      hist1 = np.zeros((np.max(res)/sh[1])+3, dtype=int)
      hist0 = np.zeros((np.max(res)/sh[1])+3, dtype=int)

      

      for i in range(len(res)):
         if i%2 == 0:
            ph = ((res[i]-1)/float(sh[1]))
            hist1[ph] += 1
         else:
            ph = (res[i]/float(sh[1]))
            hist0[ph] += 1

      fig = plt.figure()
      ax = fig.add_subplot(212, axisbg='darkslategray')
      
      ax.plot(np.arange(len(hist1)), hist1, '-o', color='y')
      ax = fig.add_subplot(211, axisbg='darkslategray')

      ax.plot(np.arange(len(hist0)), hist0, '-o', color='y')

      #ax.set_xlim(0, 1+(np.max(res)/sh[1]))
      ax.grid(True)



      plt.show()



      sys.exit()

      hz = 0.5
      fpm = 1000 / hz
         
         

      activity = []
      for i in range((zerange/2)):
         if i%fpm == 0:
            w = i
            e = w + 1000
         if i >= w and i <= e:
            activity = np.append(activity, 1)
         else:
            activity = np.append(activity, 0)


      fig = plt.figure()
      ax = fig.add_subplot(212)
   
      ax.set_title('Image')
      ax.set_xlabel("Time (ms)")
      ax.set_autoscale_on(False)
      ax.set_ylim(0,1.1)
      ax.set_xlim(0, len(activity))
      ax.plot(np.arange(len(activity)), activity, color='y', ls = '-')


         
         #fig = plt.figure()
      ax = fig.add_subplot(211)
      ax.set_title("test")
      ax.set_ylabel("Avg Firing Rate for A1")
      ax.plot(np.arange(len(A1q)), A1q, color=cm.Paired(0.06) , ls = '-')
      #ax.plot(np.arange(len(A2q)), A2q, color=cm.Paired(0.12) , ls = '-')
      #ax.plot(np.arange(len(A3q)), A3q, color=cm.Paired(0.18) , ls = '-')
      #ax.plot(np.arange(len(A4q)), A4q, color=cm.Paired(0.24) , ls = '-')
      #ax.plot(np.arange(len(A5q)), A5q, color=cm.Paired(0.30) , ls = '-')
      #ax.plot(np.arange(len(A6q)), A6q, color=cm.Paired(0.36) , ls = '-')
      #ax.plot(np.arange(len(A7q)), A7q, color=cm.Paired(0.42) , ls = '-')
      #ax.plot(np.arange(len(A8q)), A8q, color=cm.Paired(0.48) , ls = '-')
      #ax.plot(np.arange(len(A9q)), A9q, color=cm.Paired(0.54) , ls = '-')
      #ax.plot(np.arange(len(A10q)), A10q, color=cm.Paired(0.60) , ls = '-')
      #ax.plot(np.arange(len(A11q)), A11q, color=cm.Paired(0.66) , ls = '-')
      #ax.plot(np.arange(len(A12q)), A12q, color=cm.Paired(0.72) , ls = '-')
      #ax.plot(np.arange(len(A13q)), A13q, color=cm.Paired(0.78) , ls = '-')
      #ax.plot(np.arange(len(A14q)), A14q, color=cm.Paired(0.84) , ls = '-')
      #ax.plot(np.arange(len(A15q)), A15q, color=cm.Paired(0.90) , ls = '-')
      #ax.plot(np.arange(len(A16q)), A16q, color=cm.Paired(0.96) , ls = '-')


      plt.show() 


      sys.exit()
      if 1 == 1:
         kd = []
         AW = AW.reshape(lenofb, 1)
         AWO = AWO.reshape(lenofb, 1)
         count = 0

         for k in range(w.numPatches):
            p = w.next_patch()
            pO = wO.next_patch()
            kx = conv.kyPos(k, nx, ny, nf)
            ky = conv.kyPos(k, nx, ny, nf)
            if len(p) != nxp * nyp:
               continue

            #print "p = ", p

            count += 1
            #print "count = ", count
            if AW[k] == 1:
               if len(kd) == 0:
                  don = p
                  doff = pO
                  kd = np.append(don, doff)
               else:
                  don = p
                  doff = pO
                  e = np.append(don, doff)
                  kd = np.vstack((kd, e))               
               p = np.reshape(p, (nxp, nyp))
               pO = np.reshape(pO, (nxp, nyp))


            else:
               p = d
               pO = d
            #print "post p", p
            x = space + (space + nxp) * (k % nx)
            y = space + (space + nyp) * (k / nx)

            im[y:y+nyp, x:x+nxp] = p
            im2[y:y+nyp, x:x+nxp] = pO

         k = 16
         wd = sp.whiten(kd)
         result = sp.kmeans2(wd, k)
         cluster = result[1]



         nx_im5 = 2 * (nxp + space) + space
         ny_im5 = k * (nyp + space) + space
         im5 = np.zeros((nx_im5, ny_im5))
         im5[:,:] = (w.max - w.min) / 2.

         b = result[0]
         c = np.hsplit(b, 2)
         con = c[0]
         coff = c[1]

         for i in range(k):
            d = con[i].reshape(nxp, nyp)

            
            x = space + (space + nxp) * (i % k)
            y = space + (space + nyp) * (i / k)

            im5[y:y+nyp, x:x+nxp] = d

         for i in range(k):
            e = coff[i].reshape(nxp, nyp)
            i = i + k
            x = space + (space + nxp) * (i % k)
            y = space + (space + nyp) * (i / k)

            im5[y:y+nyp, x:x+nxp] = e
            


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
         acount = len(kd)

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


         h = [count1, count2, count3, count4, count5, count6, count7, count8, count9, count10, count11, count12, count13, count14, count15, count16, count18]
         h2 = [0, count1, count2, count3, count4, count5, count6, count7, count8, count9, count10, count11, count12, count13, count14, count15, count16, count18] 

         fig4 = plt.figure()
         ax4 = fig4.add_subplot(111,  axisbg='darkslategray')
         loc = np.array(range(len(h)))+0.5
         width = 1.0
         ax4.bar(loc, h, width=width, bottom=0, color='y')
         ax4.plot(np.arange(len(h2)), h2, ls = '-', marker = 'o', color='y')
         ax4.set_title("Number of Neurons that Respond to Higher than .5 max firing rate")
         ax4.set_ylabel("Number of Neurons")
         ax4.set_xlabel("Number of Presented Lines")


         fig = plt.figure()
         ax = fig.add_subplot(1,1,1)

         ax.set_xlabel('1=%1.0i 2=%1.0i 3=%1.0i 4=%1.0i 5=%1.0i 6=%1.0i 7=%1.0i 8%1.0i\n 9=%1.0i 10=%1.0i 11=%1.0i 12=%1.0i 13=%1.0i 14=%1.0i 15=%1.0i 16=%1.0i none=%1.0i' %(count1, count2, count3, count4, count5, count6, count7, count8, count9, count10, count11, count12, count13, count14, count15, count16, count18))
         ax.set_ylabel('Ky GLOBAL')
         ax.set_title('Activity: min=%1.1f, max=%1.1f time=%d' %(0, 8, a1.time))
      #ax.format_coord = format_coord
         ax.imshow(AF, cmap=cm.binary, interpolation='nearest', vmin=0., vmax=1)

         ax.text(140.0, 0.0, "How Many Above Half of Max") 
         ax.text(140.0, 5.0, "1", backgroundcolor = cm.binary(0.0))
         ax.text(140.0, 10.0, "2", backgroundcolor = cm.binary(0.06))
         ax.text(140.0, 15.0, "3", backgroundcolor = cm.binary(0.12))
         ax.text(140.0, 20.0, "4", backgroundcolor = cm.binary(0.18))
         ax.text(140.0, 25.0, "5", backgroundcolor = cm.binary(0.24))
         ax.text(140.0, 30.0, "6", backgroundcolor = cm.binary(0.30))
         ax.text(140.0, 35.0, "7", backgroundcolor = cm.binary(0.36))
         ax.text(140.0, 40.0, "8", backgroundcolor = cm.binary(0.42))
         ax.text(140.0, 45.0, "9", backgroundcolor = cm.binary(0.48))
         ax.text(140.0, 50.0, "10", backgroundcolor = cm.binary(0.54))
         ax.text(140.0, 55.0, "11", backgroundcolor = cm.binary(0.60))
         ax.text(140.0, 60.0, "12", backgroundcolor = cm.binary(0.66))
         ax.text(140.0, 66.0, "13", backgroundcolor = cm.binary(0.72))
         ax.text(140.0, 70.0, "14", backgroundcolor = cm.binary(0.78))
         ax.text(140.0, 75.0, "15", backgroundcolor = cm.binary(0.84))
         ax.text(140.0, 80.0, "16", backgroundcolor = cm.binary(0.9))

         ax.text(140.0, 85.0, "nothing", color = 'w', backgroundcolor = cm.binary(1.0))


         #fig2 = plt.figure()
         #ax2 = fig2.add_subplot(111)
         #ax2.set_xlabel('Kx GLOBAL')
         #ax2.set_ylabel('Ky GLOBAL')
         #ax2.set_title('Weight On Patches')
         #ax2.format_coord = format_coord

         #ax2.imshow(im, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)

         #fig3 = plt.figure()
         #ax3 = fig3.add_subplot(111)
         #ax3.set_xlabel('Kx GLOBAL')
         #ax3.set_ylabel('Ky GLOBAL')
         #ax3.set_title('Weight Off Patches')
         #ax3.format_coord = format_coord

         #ax3.imshow(im2, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)







         fig = plt.figure()
         ax = fig.add_subplot(111)
   
         textx = (-7/16.0) * k
         texty = (10/16.0) * k
   
         ax.set_title('On and Off K-means')
         ax.set_axis_off()
         ax.text(textx, texty,'ON\n\nOff', fontsize='xx-large', rotation='horizontal') 
         ax.text( -5, 12, "Percent %.2f   %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f" %(kcountper1,  kcountper2,  kcountper3, kcountper4, kcountper5, kcountper6, kcountper7, kcountper8, kcountper9, kcountper10, kcountper11, kcountper12, kcountper13, kcountper14, kcountper15, kcountper16), fontsize='large', rotation='horizontal')
         ax.text(-4, 14, "Patch   1      2       3       4       5       6       7       8       9      10      11     12     13     14     15     16", fontsize='x-large', rotation='horizontal')

         ax.imshow(im5, cmap=cm.jet, interpolation='nearest', vmin=w.min, vmax=w.max)





         plt.show()

#end fig loop

   sys.exit()

