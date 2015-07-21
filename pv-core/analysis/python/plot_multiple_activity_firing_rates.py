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
begin = 0
end = 10000
step = 1000
vmax = 100.0 # Hz

if len(sys.argv) < 22:
   print "usage: plot_avg_activity filename1, filename2, filename3, filename4, filename5, filename6, filename7, filename8, filename9, filename10, filename11, filename12, filename13, filename14, filename15, filename16 [end_time step_time begin_time], test filename, On-weigh filename, Off-weight filename"
   sys.exit()

#if len(sys.argv) >= 6:
#   vmax = float(sys.argv[5])

print "(begin, end, step, max) == ", begin, end, step, vmax

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
zerange = int(sys.argv[17])
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

pa = []

for endtest in range(begintest+steptest, endtest, steptest):
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


im2 = np.zeros((nx_im, ny_im))
im2[:,:] = (w.max - w.min) / 2.


print "avg = ", avg
print "median = ", median
#a2.rewind()
co = 0
for g in range(2):
   if g == 0:
      for end in range(begin+step, end, step):
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
         AF = A1

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
      numofsteps = 10
      for k in range(zerange):    ####### range(step)
         if k%1000 == 0:
            print "at ", k
         test = []
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
         for i in range(lenofo):
            for j in range(lenofo):
               if AWO[i, j] == 1:
                  if AW[i, j] == 1:
                     t = 0
                     for h in range(len(A1A)):
                        if A1A[h] == ((lenofo * i) + j):
                           t = 1
                           #SUMAW[i, j] += 1
                     if t == 1:
                        test = np.append(test,1)
                        #print "its echo lasts longer"
                     else:
                        test = np.append(test, 0)

         #if np.sum(test) > 0:
         #   print "test = ", test
         #   print "sum = ", sum(test)
         if k == 0:
            testq = np.sum(test)
         else:
            testq = np.append(testq, np.sum(test))
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
         
      testq = (testq / len(test)) / (numofsteps / 100.0)
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
      ax.plot(np.arange(len(testq)), testq, color='y', ls = '-')
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

