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
#zetest = rs.PVReadSparse(sys.argv[21], extended)
w =  rw.PVReadWeights(sys.argv[21])
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
margin = 10

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
A1pos2 = np.array([0,0])
countpos = 0
countpos2 = 0

print "avg = ", avg
print "median = ", median
#a2.rewind()
co = 0
for g in range(2):
   if g == 0:
      for end in range(begin+step, step+1, step):
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
         countpos = 0

         lenofo = len(A1)
         lenofb = lenofo * lenofo
         beingplotted = []

         wex = 16
         dex = np.array([])

         for i in range(lenofo):
            for j in range(lenofo):
               #print A1[i, j]
               check = [A1[i,j], A2[i,j], A3[i,j], A4[i,j], A5[i,j], A6[i,j], A7[i,j], A8[i,j], A9[i,j], A10[i,j], A11[i,j], A12[i,j], A13[i,j], A14[i,j], A15[i,j], A16[i,j]] 
               checksort = np.sort(check)
               wcount = 0
               for v in range(len(check)):
                  if check[v]==checksort[-wex] and wcount==0:
                     pex = v
                     wcount += 1
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
                     countnum += 1
                     if i > margin and i < (w.nx - margin):
                        if j > margin and j < (w.ny - margin):
                           if countpos == 0:
                              A1pos = [i, j]
                           else:
                              A1pos = np.vstack((A1pos, [i, j]))
                           if countpos == 0:
                              dex = pex
                           else:
                              dex = np.append(pex, dex)
                           countpos+=1

   
               elif co == 2:
                  AF[i, j] = 0.06
                  count2 += 1
                  AWO[i, j] = 2.0


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

      alen = np.shape(A1pos)[0]

      print "pos shape = ", np.shape(A1pos)
      print "A1pos = ", A1pos

      print "dex = ", dex
      print np.shape(dex)


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
      #print A1pos
      #print np.shape(A1pos)
      #A1pos = np.vstack((A1pos, [0, 0]))


      for k in range(zerange):    ####### range(step)
         if k%1000 == 0:
            print "at ", k
         A1t = []
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
         A2A = a2.next_record()
         A3A = a3.next_record()
         A4A = a4.next_record()
         A5A = a5.next_record()
         A6A = a6.next_record()
         A7A = a7.next_record()
         A8A = a8.next_record()
         A9A = a9.next_record()
         A10A = a10.next_record()
         A11A = a11.next_record()
         A12A = a12.next_record()
         A13A = a13.next_record()
         A14A = a14.next_record()
         A15A = a15.next_record()
         A16A = a16.next_record()
         A1t = np.zeros((1, np.shape(A1pos)[0]))
         A1t2 = np.zeros((1, np.shape(A1pos)[0]))



#####
         for g in range(np.shape(A1pos)[0]):
            w = A1pos[g]
            i = w[0]
            j = w[1]
            for h in range(len(A1A)):
               if A1A[h] == ((lenofo * i) + j):
                  A1t[0, g] += 1

            if dex[g] == 1:
               for h in range(len(A2A)):
                  if A2A[h] == ((lenofo * i) + j):
                     A1t2[0, g] += 1
            if dex[g] == 2:
               for h in range(len(A3A)):
                  if A3A[h] == ((lenofo * i) + j):
                     A1t2[0, g] += 1
            if dex[g] == 3:
               for h in range(len(A4A)):
                  if A4A[h] == ((lenofo * i) + j):
                     A1t2[0, g] += 1
            if dex[g] == 4:
               for h in range(len(A5A)):
                  if A5A[h] == ((lenofo * i) + j):
                     A1t2[0, g] += 1
            if dex[g] == 5:
               for h in range(len(A6A)):
                  if A6A[h] == ((lenofo * i) + j):
                     A1t2[0, g] += 1
            if dex[g] == 6:
               for h in range(len(A7A)):
                  if A7A[h] == ((lenofo * i) + j):
                     A1t2[0, g] += 1
            if dex[g] == 7:
               for h in range(len(A8A)):
                  if A8A[h] == ((lenofo * i) + j):
                     A1t2[0, g] += 1
            if dex[g] == 8:
               for h in range(len(A9A)):
                  if A9A[h] == ((lenofo * i) + j):
                     A1t2[0, g] += 1
            if dex[g] == 9:
               for h in range(len(A10A)):
                  if A10A[h] == ((lenofo * i) + j):
                     A1t2[0, g] += 1
            if dex[g] == 10:
               for h in range(len(A11A)):
                  if A11A[h] == ((lenofo * i) + j):
                     A1t2[0, g] += 1
            if dex[g] == 11:
               for h in range(len(A12A)):
                  if A12A[h] == ((lenofo * i) + j):
                     A1t2[0, g] += 1
            if dex[g] == 12:
               for h in range(len(A13A)):
                  if A13A[h] == ((lenofo * i) + j):
                     A1t2[0, g] += 1
            if dex[g] == 13:
               for h in range(len(A14A)):
                  if A14A[h] == ((lenofo * i) + j):
                     A1t2[0, g] += 1
            if dex[g] == 14:
               for h in range(len(A15A)):
                  if A15A[h] == ((lenofo * i) + j):
                     A1t2[0, g] += 1
            if dex[g] == 15:
               for h in range(len(A16A)):
                  if A16A[h] == ((lenofo * i) + j):
                     A1t2[0, g] += 1



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
                        A13t = np.append(A13t,1)
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
                        A15t = np.append(A15t,1)
                     else:
                        A15t = np.append(A15t, 0)
                  if AW[i, j] == 16:
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

         if k >= (numofsteps*d) and k < ((numofsteps * d) + numofsteps):
            if k == (numofsteps * d):
               A1p2 = A1t2
               thecount+=1
            else:
               A1p2 = np.vstack((A1p2,A1t2))
               thecount+=1
         if k == (numofsteps-1):
            A1q2 = A1p2.sum(axis=0) 
         if k == ((numofsteps*d) + (numofsteps-1)) and k != (numofsteps-1):
            A1q2 = np.vstack((A1q2, A1p2.sum(axis=0)))



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

      #A2q = (A2q / len(A2t)) / (numofsteps / 100.0)
      #A3q = (A3q / len(A3t)) / (numofsteps / 100.0)
      #A4q = (A4q / len(A4t)) / (numofsteps / 100.0)
      #A5q = (A5q / len(A5t)) / (numofsteps / 100.0)
      #A6q = (A6q / len(A6t)) / (numofsteps / 100.0)
      #A7q = (A7q / len(A7t)) / (numofsteps / 100.0)
      #A8q = (A8q / len(A8t)) / (numofsteps / 100.0)
      #A9q = (A9q / len(A9t)) / (numofsteps / 100.0)
      #A10q = (A10q / len(A10t)) / (numofsteps / 100.0)
      #A11q = (A11q / len(A11t)) / (numofsteps / 100.0)
      #A12q = (A12q / len(A12t)) / (numofsteps / 100.0)
      #A13q = (A13q / len(A13t)) / (numofsteps / 100.0)
      #A14q = (A14q / len(A14t)) / (numofsteps / 100.0)
      #A15q = (A15q / len(A15t)) / (numofsteps / 100.0)
      #A16q = (A16q / len(A16t)) / (numofsteps / 100.0)

      sh = np.shape(A1q)
      sh2 = np.shape(A1q2)

      print "1st sh = ", sh

      for i in range((sh[0]/2)):
         A1q = np.delete(A1q, (i+1), axis=0)
      for i in range((sh2[0]/2)):
         A1q2 = np.delete(A1q2, (i+1), axis=0)
 
      sh = np.shape(A1q)

      print "shape = ", sh

      for i in range(sh[0]):
         if i == 0:
            a = np.array([1])
            a2 = np.array([0])
         else:
            a = np.vstack((a, 1))
            a2 = np.vstack((a2, 0))

      #print "A1q shape = ", np.shape(A1q)
      #print "a shape = ", np.shape(a)
      #print "A1q = ", A1q
      #print "A1q2 = ", A1q2

      h1 = A1q
      h2 = A1q2
      aa = np.vstack((a, a2))
      A1q = np.vstack((A1q, A1q2))

      A1q = np.insert(A1q, [0], aa, axis=1)
      print A1q

      np.savetxt("roc-%i.txt" %(wex), A1q, fmt='%d', delimiter = ';')   


      res = np.average(h1, axis=1)
      res2 = np.average(h2, axis=1)
      print res
      print res2
      hist1 = np.histogram(res, np.arange(np.max(res) + 2))
      hist2 = np.histogram(res2, np.arange(np.max(res2) + 2))

      print hist1
      print hist2
      h1a = hist1[0]
      h1b = hist1[1]
      h2a = hist2[0]
      h2b = hist2[1]
      print
      print
      print h1a
      #print np.shape(h1a)
      #print h1b
      #print np.shape(h1b)
      print h2a
      #print np.shape(h2a)
      #print h2b
      #print np.shape(h2b)
      print np.arange(len(h1a))



      fig = plt.figure()
      ax = fig.add_subplot(111)
      
      ax.plot(np.arange(len(h1a)), h1a, '-o', color='b')
      ax.plot(np.arange(len(h2a)), h2a, '--o', color='b')

      #ax.plot(np.arange(len(hist0)), hist0, 'o', color='y')

      ax.set_xlabel('CLIQUE BINS')
      ax.set_ylabel('COUNT')
      ax.set_title('Clique Histogram')
      ax.grid(True)





      plt.show()



      sys.exit()
