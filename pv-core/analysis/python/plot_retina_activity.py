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


if len(sys.argv) < 4:
   print "usage: plot_retina_activity activity-filename end step begin"

extended = False

a1 = rs.PVReadSparse(sys.argv[1], extended)
end = int(sys.argv[2])
step = int(sys.argv[3])
begin = int(sys.argv[4])

print a1
print end
print step
print begin


activity = []
count = 0

counta=0
for i in range(end):
   A=a1.next_activity()
   for g in range(len(A)):
      for h in range(len(A)):
         if g == 13 & h == 13:
            activity = np.append(activity, A[g, h])
            if A[g,h] == 1:
               counta+=1
            #print activity
print counta
fig = plt.figure()
ax = fig.add_subplot(111)
   
#ax.set_title('Time = %1.1f' %(atime))

ax.set_autoscale_on(False)
ax.set_ylim(0,1.2)
ax.set_xlim(0, len(activity))
ax.plot(np.arange(len(activity)), activity, color='y', ls = '-')

plt.show() 

sys.exit()



for end in range(begin+step, end, step):
   A = a1.avg_activity(begin, end)


   for g in range(len(A)):
      for h in range(len(A)):
         if g == 16 & h == 16:
            activity = np.append(activity, A[g, h])
   count += 1   

   print "time = ", a1.time
   atime = a1.time
   print count
   #sys.exit()
   fig = plt.figure()
   ax = fig.add_subplot(111)
   
   ax.set_title('Time = %1.1f' %(atime))

   #ax.set_autoscale_on(False)
   #ax.set_ylim(0,)
   #ax.set_xlim(0, len(activity))
   ax.plot(np.arange(len(activity)), activity, color='y', ls = '-')

   plt.show() 

   sys.exit()
