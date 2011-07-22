
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

if len(sys.argv) < 26:
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
a17 = rs.PVReadSparse(sys.argv[17], extended)
a18 = rs.PVReadSparse(sys.argv[18], extended)
a19 = rs.PVReadSparse(sys.argv[19], extended)
a20 = rs.PVReadSparse(sys.argv[20], extended)

end = int(sys.argv[21])
step = int(sys.argv[22])
begin = int(sys.argv[23])
endtest = end
steptest = step
begintest = begin
atest = rs.PVReadSparse(sys.argv[24], extended)
#zetest = rs.PVReadSparse(sys.argv[21], extended)
w =  rw.PVReadWeights(sys.argv[25])
wO = rw.PVReadWeights(sys.argv[26])
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
AWmin = np.zeros((lenofo, lenofo))

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
minpos = np.array([0,0])
minlist = np.array([0,0])
countpos = 0

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
         A17 = a17.avg_activity(begin, end)
         A18 = a18.avg_activity(begin, end)
         A19 = a19.avg_activity(begin, end)
         A20 = a20.avg_activity(begin, end)

         AF = np.zeros((lenofo, lenofo))
         countpos = 0

         lenofo = len(A1)
         lenofb = lenofo * lenofo
         beingplotted = []
         for i in range(lenofo):
            for j in range(lenofo):
               #print A1[i, j]
               check = [A1[i,j], A2[i,j], A3[i,j], A4[i,j], A5[i,j], A6[i,j], A7[i,j], A8[i,j], A9[i,j], A10[i,j], A11[i,j], A12[i,j], A13[i,j], A14[i,j], A15[i,j], A16[i,j], A17[i,j], A18[i,j], A19[i,j], A20[i,j]] 

               checkmax = np.max(check)
               checkmin = np.min(check)
               wheremax = np.argmax(check)
               wheremin = np.argmin(check)
               #print "wheremin = ", wheremin

               half = checkmax / 2.0
               sort = np.sort(check)
               co = 0
               if wheremin == 0:
                  AWmin[i, j] = 1
               if wheremin == 1:
                  AWmin[i, j] = 2
               if wheremin == 2:
                  AWmin[i, j] = 3
               if wheremin == 3:
                  AWmin[i, j] = 4
               if wheremin == 4:
                  AWmin[i, j] = 5
               if wheremin == 5:
                  AWmin[i, j] = 6
               if wheremin == 6:
                  AWmin[i, j] = 7
               if wheremin == 7:
                  AWmin[i, j] = 8
               if wheremin == 8:
                  AWmin[i, j] = 9
               if wheremin == 9:
                  AWmin[i, j] = 10
               if wheremin == 10:
                  AWmin[i, j] = 11
               if wheremin == 11:
                  AWmin[i, j] = 12
               if wheremin == 12:
                  AWmin[i, j] = 13
               if wheremin == 13:
                  AWmin[i, j] = 14
               if wheremin == 14:
                  AWmin[i, j] = 15
               if wheremin == 15:
                  AWmin[i, j] = 16
               if wheremin == 16:
                  AWmin[i, j] = 17
               if wheremin == 17:
                  AWmin[i, j] = 18
               if wheremin == 18:
                  AWmin[i, j] = 19
               if wheremin == 19:
                  AWmin[i, j] = 20



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
               if wheremax == 16:
                  AW[i, j] = 17
               if wheremax == 17:
                  AW[i, j] = 18
               if wheremax == 18:
                  AW[i, j] = 19
               if wheremax == 19:
                  AW[i, j] = 20
               

               #print AF[i, j]
               #print "check = ", sort
               #print "half = ", half
               countnum += 1
               if i > margin and i < (w.nx - margin):
                  if j > margin and j < (w.ny - margin):
                           #print "QUICKLY"
                     if countpos == 0:
                        A1pos = [i, j]
                        minpos = [i, j]
                        A1list = [wheremax]
                        minlist = [wheremin]
                     else:
                        A1pos = np.vstack((A1pos, [i, j]))
                        minpos = np.vstack((minpos, [i,j]))
                        A1list = np.vstack((A1list, wheremax))
                        minlist = np.vstack((minlist, wheremin))
                     countpos+=1

      print "np.shape = ", np.shape(minlist)
      print "minlist = ", minlist
      print "minpos = ", minpos



      print "pos shape = ", np.shape(A1pos)
      print "A1pos = ", A1pos

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
      a17.rewind()
      a18.rewind()
      a19.rewind()
      a20.rewind()

      print "after rewind"

      countg = 0
      testgraph = []
      test = []
      numofsteps = 500
      #print A1pos
      #print np.shape(A1pos)
      #A1pos = np.vstack((A1pos, [0, 0]))
      sail = 0
      for k in range(zerange):    ####### range(step)
         if k%100 == 0:
            print "at ", k
         if k%1000 == 0 and sail > 0:
            print "at ", k
            print "Done"
            sys.exit()
         sail+=1
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
         A17t = []
         A18t = []
         A19t = []
         A20t = []


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
         A17A = a17.next_record()
         A18A = a18.next_record()
         A19A = a19.next_record()
         A20A = a20.next_record()


         A1t = np.zeros((1, np.shape(A1pos)[0]))
         mint = np.zeros((1, np.shape(A1pos)[0]))

         print "got hur now"
#####
         for g in range(np.shape(A1pos)[0]):
            w = A1pos[g]
            i = w[0]
            j = w[1]

            if A1list[g] == 0:
               for h in range(len(A1A)):
                  if A1A[h] == ((lenofo * i) + j):
                     A1t[0, g] += 1
            if A1list[g] == 1:
               for h in range(len(A2A)):
                  if A2A[h] == ((lenofo * i) + j):
                     A1t[0, g] += 1
            if A1list[g] == 2:
               for h in range(len(A3A)):
                  if A3A[h] == ((lenofo * i) + j):
                     A1t[0, g] += 1
            if A1list[g] == 3:
               for h in range(len(A4A)):
                  if A4A[h] == ((lenofo * i) + j):
                     A1t[0, g] += 1
            if A1list[g] == 4:
               for h in range(len(A5A)):
                  if A5A[h] == ((lenofo * i) + j):
                     A1t[0, g] += 1
            if A1list[g] == 5:
               for h in range(len(A6A)):
                  if A6A[h] == ((lenofo * i) + j):
                     A1t[0, g] += 1
            if A1list[g] == 6:
               for h in range(len(A7A)):
                  if A7A[h] == ((lenofo * i) + j):
                     A1t[0, g] += 1
            if A1list[g] == 7:
               for h in range(len(A8A)):
                  if A8A[h] == ((lenofo * i) + j):
                     A1t[0, g] += 1
            if A1list[g] == 8:
               for h in range(len(A9A)):
                  if A9A[h] == ((lenofo * i) + j):
                     A1t[0, g] += 1
            if A1list[g] == 9:
               for h in range(len(A10A)):
                  if A10A[h] == ((lenofo * i) + j):
                     A1t[0, g] += 1
            if A1list[g] == 10:
               for h in range(len(A11A)):
                  if A11A[h] == ((lenofo * i) + j):
                     A1t[0, g] += 1
            if A1list[g] == 11:
               for h in range(len(A12A)):
                  if A12A[h] == ((lenofo * i) + j):
                     A1t[0, g] += 1
            if A1list[g] == 12:
               for h in range(len(A13A)):
                  if A13A[h] == ((lenofo * i) + j):
                     A1t[0, g] += 1
            if A1list[g] == 13:
               for h in range(len(A14A)):
                  if A14A[h] == ((lenofo * i) + j):
                     A1t[0, g] += 1


         print "just a bit later"
         if 1 == 1:
            for x in range(len(minlist)):
               w = minpos[x]
               i = w[0]
               j = w[1]

               if minlist[x] == 0:
                  for y in range(len(A1A)):
                     if A1A[y] == ((lenofo * i) + j):
                        mint[0, x] += 1
               if minlist[x] == 1:
                  for y in range(len(A2A)):
                     if A2A[y] == ((lenofo * i) + j):
                        mint[0, x] += 1
               if minlist[x] == 2:
                  for y in range(len(A3A)):
                     if A3A[y] == ((lenofo * i) + j):
                        mint[0, x] += 1
               if minlist[x] == 3:
                  for y in range(len(A4A)):
                     if A4A[y] == ((lenofo * i) + j):
                        mint[0, x] += 1
               if minlist[x] == 4:
                  for y in range(len(A5A)):
                     if A5A[y] == ((lenofo * i) + j):
                        mint[0, x] += 1
               if minlist[x] == 5:
                  for y in range(len(A6A)):
                     if A6A[y] == ((lenofo * i) + j):
                        mint[0, x] += 1
               if minlist[x] == 6:
                  for y in range(len(A7A)):
                     if A7A[y] == ((lenofo * i) + j):
                        mint[0, x] += 1
               if minlist[x] == 7:
                  for y in range(len(A8A)):
                     if A8A[y] == ((lenofo * i) + j):
                        mint[0, x] += 1
               if minlist[x] == 8:
                  for y in range(len(A9A)):
                     if A9A[y] == ((lenofo * i) + j):
                        mint[0, x] += 1
               if minlist[x] == 9:
                  for y in range(len(A10A)):
                     if A10A[y] == ((lenofo * i) + j):
                        mint[0, x] += 1
               if minlist[x] == 10:
                  for y in range(len(A11A)):
                     if A11A[y] == ((lenofo * i) + j):
                        mint[0, x] += 1
               if minlist[x] == 11:
                  for y in range(len(A12A)):
                     if A12A[y] == ((lenofo * i) + j):
                        mint[0, x] += 1
               if minlist[x] == 12:
                  for y in range(len(A13A)):
                     if A13A[y] == ((lenofo * i) + j):
                        mint[0, x] += 1
               if minlist[x] == 13:
                  for y in range(len(A14A)):
                     if A14A[y] == ((lenofo * i) + j):
                        mint[0, x] += 1



         print "mortal kombat!!!"

         #if np.sum(test) > 0:
         #   print "test = ", test
         #   print "sum = ", sum(test)
         #print "A1t = ", A1t
         #print "mint = ", mint


         d = k / numofsteps
         #print
         #print "A1t = ", A1t
         #print np.shape(A1t)

         if k >= (numofsteps*d) and k < ((numofsteps * d) + numofsteps):
            if k == (numofsteps * d):
               A1p = A1t
               minp = mint
               thecount+=1
            else:
               A1p = np.vstack((A1p,A1t))
               minp = np.vstack((minp, mint))
               #print "A1p = ", A1p
               #print "minp = ", minp
               #print
               thecount+=1
         if k == (numofsteps-1):
            A1q = A1p.sum(axis=0) 
            minq = minp.sum(axis=0)
         if k == ((numofsteps*d) + (numofsteps-1)) and k != (numofsteps-1):
            A1q = np.vstack((A1q, A1p.sum(axis=0)))
            minq = np.vstack((minq, minp.sum(axis=0)))
            #print "A1q = ", A1q
            #print "minq = ", minq



      print "a1q = ", A1q
      print "minq = ", minq



      sh = np.shape(A1q)
      minsh = np.shape(minq)
      print "shape = ", sh
      print "minshape = ", minsh




      for i in range(sh[0]):
         z = i%2
         if i == 0:
            a = np.array([1])
            b = np.array([0])
            A1qf = np.array(A1q[i])
            minqf = np.array(minq[i])
         if i != 0 and (z==0):
            a = np.vstack((a, 1))
            b = np.vstack((b, 0))
            A1qf = np.vstack((A1qf, A1q[i]))
            minqf = np.vstack((minqf, minq[i]))
         #if z==4 or z==5 or z==6 or z==7 and i!= 0:
         #   a = np.vstack((a,0))
      #print "A1q shape = ", np.shape(A1q)
      #print "a shape = ", np.shape(a)

      print A1qf
      print minqf


      res = np.sum(A1q, axis=1)
      minres = np.sum(minq, axis=1)
      hist1 = np.zeros((np.max(res)/sh[1])+3, dtype=int)
      hist2 = np.zeros((np.max(minres)/minsh[1])+3, dtype=int)

      for i in range(len(res)):
         z = i%2
         if z==0:
            ph = ((res[i])/float(sh[1]))
            hist1[ph] += 1
            minph = (minres[i]/float(minsh[1]))
            hist2[minph] += 1
#         if z==4 or z==5 or z==6 or z==7:
#            ph = (res[i]/float(sh[1]))
#            hist2[ph] += 1

      a = np.vstack((a, b))
      A1qf = np.vstack((A1qf, minqf))


      A1qf = np.insert(A1qf, [0], a, axis=1)


      np.savetxt("roc-info.txt", A1qf, fmt='%d', delimiter = ';')        

      sys.exit()
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
