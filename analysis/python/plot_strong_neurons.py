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

extended = False


a1 = rs.PVReadSparse(sys.argv[1], extended)

end = int(sys.argv[2])
step = int(sys.argv[3])
begin = int(sys.argv[4])
endtest = int(sys.argv[2])
steptest = int(sys.argv[3])
begintest = int(sys.argv[4])
atest = rs.PVReadSparse(sys.argv[5], extended)
zerange = end


where = []
count = 0

for endtest in range(begintest+steptest, steptest+1, steptest):
   Atest = atest.avg_activity(begintest, endtest)
   nmax = np.max(Atest)
   lenofo = len(Atest)
   for i in range(lenofo):
      for j in range(lenofo):
         if Atest[i,j] > (0.7 * nmax):
            if count == 0:
               where = [i,j]
            else:
               where = np.vstack((where, [i, j]))
            count+=1

print "shape of where = ", np.shape(where)
print np.shape(where)[0]

a1.rewind()

A1t = np.zeros((1, np.shape(where)[0]))

for k in range(zerange):

   A1 = a1.next_record()
   A1t = np.zeros((1, np.shape(where)[0]))


   for g in range(np.shape(where)[0]):
      w = where[g]
      i = w[0]
      j = w[1]
   for h in range(len(A1)):
      if A1[h] == ((lenofo * i) + j):
         A1t[0,g] = 1
   if k == 0:
      A1p = np.average(A1t)
   else:
      A1p = np.append(A1p, np.average(A1t))


fig = plt.figure()
ax = fig.add_subplot(111, axisbg='darkslategray')
      
ax.plot(np.arange(len(A1p)), A1p, '-o', color='y')

ax.set_xlabel('time (0.5 ms)   num of neurons =  %d' %(np.shape(where)[0]))
ax.set_ylabel('Avg Firing Rate')
ax.set_title('Average High Activity')

ax.grid(True)


plt.show()
