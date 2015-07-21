
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

if len(sys.argv) < 19:
   print "usage: plot_avg_activity filename 1-32, [end_time step_time begin_time], test filename, On-weigh filename, Off-weight filename"
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
endtest = step
steptest = step
begintest = begin
#zetest = rs.PVReadSparse(sys.argv[21], extended)


zerange = end
margin = 15

pa = []


print "(begin, end, step, max) == ", begin, end, step, vmax

space = 1

d = np.zeros((5,5))
coord = 1


res = 0
rmax = 0
rmin = 0
count = 0

#print "avg = ", avg
#print "median = ", median
#a2.rewind()
co = 0

ecount = 0

lenofo = a1.nx

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

   if ecount == 0:
      ecount+=1
      stim = np.zeros((lenofo, lenofo))



   for i in range(lenofo):
      for j in range(lenofo):
          #print A1[i, j]
          check = [A1[i,j], A2[i,j], A3[i,j], A4[i,j], A5[i,j], A6[i,j], A7[i,j], A8[i,j], A9[i,j], A10[i,j], A11[i,j], A12[i,j], A13[i,j], A14[i,j], A15[i,j], A16[i,j]]

          count += 1
          checkmax = np.max(check)
          checkmin = np.min(check)
          argmax = np.argmax(check)


          if checkmax == 0 and checkmin == 0:
             print "both equal 0"

          if argmax == 0:
             stim[i, j] = 1
          if argmax == 1:
             stim[i, j] = 2
          if argmax == 2:
             stim[i, j] = 3
          if argmax == 3:
             stim[i, j] = 4
          if argmax == 4:
             stim[i, j] = 5
          if argmax == 5:
             stim[i, j] = 6
          if argmax == 6:
             stim[i, j] = 7
          if argmax == 7:
             stim[i, j] = 8
          if argmax == 8:
             stim[i, j] = 9
          if argmax == 9:
             stim[i, j] = 10
          if argmax == 10:
             stim[i, j] = 11
          if argmax == 11:
             stim[i, j] = 12
          if argmax == 12:
             stim[i, j] = 13
          if argmax == 13:
             stim[i, j] = 14
          if argmax == 14:
             stim[i, j] = 15
          if argmax == 15:
             stim[i, j] = 16



print "stim"
print stim


#########


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

print "endtest = ", endtest

time1 = np.zeros(endtest)
time2 = np.zeros(endtest)
time3 = np.zeros(endtest)
time4 = np.zeros(endtest)
time5 = np.zeros(endtest)
time6 = np.zeros(endtest)
time7 = np.zeros(endtest)
time8 = np.zeros(endtest)
time9 = np.zeros(endtest)
time10 = np.zeros(endtest)
time11 = np.zeros(endtest)
time12 = np.zeros(endtest)
time13 = np.zeros(endtest)
time14 = np.zeros(endtest)
time15 = np.zeros(endtest)
time16 = np.zeros(endtest)

zcount = 0

for i in range(endtest):
   A1 = a1.next_activity()
   A2 = a2.next_activity()
   #A3 = a3.next_activity()
   #A4 = a4.next_activity()
   #A5 = a5.next_activity()
   #A6 = a6.next_activity()
   #A7 = a7.next_activity()
   #A8 = a8.next_activity()
   #A9 = a9.next_activity()
   #A10 = a10.next_activity()
   #A11 = a11.next_activity()
   #A12 = a12.next_activity()
   #A13 = a13.next_activity()
   #A14 = a14.next_activity()
   #A15 = a15.next_activity()
   #A16 = a16.next_activity()

   for i in range(lenofo):
      for j in range(lenofo):
         if 20 < i < 108 and 20 < j < 108 :
 
            stimvalue = stim[i, j]
         
            if stimvalue == 1:
               time1[zcount]+=A1[i, j]
            if stimvalue == 2:
               time2[zcount]+=A2[i, j]
            #if stimvalue == 13:
            #   time13[zcount]+=A13[i, j]
            #if stimvalue == 14:
            #   time14[zcount]+=A14[i, j]


         """
         if stimvalue == 3:
            time3[zcount]+=A3[i, j]
         if stimvalue == 4:
            time4[zcount]+=A4[i, j]
         if stimvalue == 5:
            time5[zcount]+=A5[i, j]
         if stimvalue == 6:
            time6[zcount]+=A6[i, j]
         if stimvalue == 7:
            time7[zcount]+=A7[i, j]
         if stimvalue == 8:
            time8[zcount]+=A8[i, j]
         if stimvalue == 9:
            time9[zcount]+=A9[i, j]
         if stimvalue == 10:
            time10[zcount]+=A10[i, j]
         if stimvalue == 11:
            time11[zcount]+=A11[i, j]
         if stimvalue == 12:
            time12[zcount]+=A12[i, j]
         if stimvalue == 15:
            time15[zcount]+=A15[i, j]
         if stimvalue == 16:
            time16[zcount]+=A16[i, j]
         """

   zcount+=1


time1 = time1[-500:]
time2 = time2[-500:]
#time13 = time13[-500:]
#time14 = time14[-500:]


print "time1 = "
print time1
print np.shape(time1)



ran = np.arange(len(time1))


fig = plt.figure()

ax = fig.add_subplot(311)
width = 1

ax.plot(ran, time1, '-o', color=cm.Blues(0.75), linewidth=5.0)
ax.plot(ran, time2, '-o', color=cm.Greens(0.75), linewidth=5.0)

#ax.plot(ran, time13, '-o', color=cm.Reds(1.0), linewidth=5.0)
#ax.plot(ran, time14, '-o', color=cm.Greys(1.0), linewidth=5.0)
#ax.plot(ran, time5, '-o', color=cm.Blues(0.6), linewidth=5.0)
#ax.plot(ran, time6, '-o', color=cm.Blues(0.5), linewidth=5.0)
#ax.plot(ran, time7, '-o', color=cm.Blues(0.4), linewidth=5.0)
#ax.plot(ran, time8, '-o', color=cm.Blues(0.3), linewidth=5.0)

ax2 = fig.add_subplot(312)
ax2.acorr(time1, usevlines=True, normed=True, maxlags=100, color=cm.Blues(0.75))
ax3 = fig.add_subplot(313)
#ax3.acorr(time1, maxlags=100, normed=True, color=cm.Greens(1.0)) 
ax3.acorr(time2, usevlines=True, normed=True, maxlags=100, color=cm.Greens(0.75))

#ax2.acorr(time13, maxlags=300, color=cm.Reds(1.0)) 
#ax2.acorr(time14, maxlags=300, color=cm.Greys(1.0)) 

#ax2.plot(ran, time9, '-o', color=cm.Greens(1.0), linewidth=5.0)
#ax2.plot(ran, time10, '-o', color=cm.Greens(0.9), linewidth=5.0)
#ax2.plot(ran, time11, '-o', color=cm.Greens(0.8), linewidth=5.0)
#ax2.plot(ran, time12, '-o', color=cm.Greens(0.7), linewidth=5.0)
#ax2.plot(ran, time13, '-o', color=cm.Greens(0.6), linewidth=5.0)
#ax2.plot(ran, time14, '-o', color=cm.Greens(0.5), linewidth=5.0)
#ax2.plot(ran, time15, '-o', color=cm.Greens(0.4), linewidth=5.0)
#ax2.plot(ran, time16, '-o', color=cm.Greens(0.3), linewidth=5.0)

plt.show()
