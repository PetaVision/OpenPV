
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

if len(sys.argv) < 22:
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
endtest = end
steptest = step
begintest = begin
atest = rs.PVReadSparse(sys.argv[20], extended)
#zetest = rs.PVReadSparse(sys.argv[21], extended)
w =  rw.PVReadWeights(sys.argv[21])
wO = rw.PVReadWeights(sys.argv[22])


zerange = end
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
d = np.zeros((5,5))
coord = 1

numpat = w.numPatches


res = 0
rmax = 0
rmin = 0
count = 0

#print "avg = ", avg
#print "median = ", median
#a2.rewind()
co = 0


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

   vmax = 0
   hmax = 0
   vdif = 0
   hdif = 0
   vcount = 0
   hcount = 0
   vd = 0
   hd = 0
   count = 0

   whis = np.zeros((16))

   plot = np.zeros((lenofo, lenofo))

   for i in range(lenofo):
      for j in range(lenofo):
          #print A1[i, j]
          checkv = [A1[i,j], A2[i,j], A3[i,j], A4[i,j], A5[i,j], A6[i,j], A7[i,j], A8[i,j]] 
          checkh = [A9[i,j], A10[i,j], A11[i,j], A12[i,j], A13[i,j], A14[i,j], A15[i,j], A16[i,j]]




          count += 1
          checkmaxv = np.max(checkv)
          checkminv = np.min(checkv)
 
          checkmaxh = np.max(checkh)
          checkminh = np.min(checkh)

          if checkmaxv == 0 and checkminv == 0:
             print "both equal 0"
          elif checkmaxh == 0 and checkminh == 0:
             print "both equal 0"

          
          if checkmaxv > checkmaxh:
             vmax += checkmaxv
             vdif += checkmaxv - checkmaxh  
             vcount += 1
             vdd = checkmaxv-checkmaxh
             if vdd < 1.0:
                vd+=1
                #print "vd = ", vd

             am = np.argmax(checkv)
             whis[am]+=1
    

          else:
             hmax += checkmaxh
             hdif += checkmaxh - checkmaxv  
             hcount += 1
             hdd = checkmaxh-checkmaxv
             if hdd < 1.0:
                hd+=1
                #print "hd = ", hd
             am = np.argmax(checkh) + 8
             whis[am]+=1

             


   vmax = vmax / vcount
   vdif = vdif / vcount
   hmax = hmax / hcount
   hdif = hdif / hcount

   print
   print "vmax = ", vmax
   print "vdif = ", vdif
   print "times vdif < 1 = ", vd
   print "vcount = ", vcount
   print
   print "hmax = ", hmax
   print "hdif = ", hdif
   print "times hdif < 1 = ", hd
   print "hcount = ", hcount
   print 
   print "vcount + hcount = ", vcount + hcount
   print "total count = ", count

   fig = plt.figure()
   ax = fig.add_subplot(111)
   loc = np.array(range(len(whis)))+0.5
   width = 1.0
   ax.set_title('Feature Histogram')
   #ax.set_xlabel('Total Number of Features = %1.0i \n ratio = %f \n percent of total = %f' %(len(where), hratio, ptotal)) 
   ax.bar(loc, whis, width=width, bottom=0, color='b')
   ax.set_xlabel("vcount = %i  hcount = %i" %(vcount, hcount))

   plt.show()
