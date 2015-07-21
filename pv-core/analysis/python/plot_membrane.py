import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import PVReadWeights as rw
import PVConversions as conv


if len(sys.argv) < 2:
   print "usage: membrane activity-filename"
   print len(sys.argv)
   sys.exit()






a = open(sys.argv[1], "r")
v   = []
t   = []
t2  = []
vth = []
act = []
ge  = []
gi  = []
gib = []
aa  = []
aa2 = []
actcount = 0


for line in a:
   a = line
   b = a.find("V=")
   h = a.find("t=")
   bth = a.find("Vth=")
   gep = a.find("G_E=")
   gip = a.find("G_I=")
   gibp = a.find("G_IB=")
   actp = a.find("a=")  

   c = a[b:].split()
   cth = a[bth:].split()
   i = a[h:].split()
   gif = a[gip:].split()
   gef = a[gep:].split()
   gibf = a[gibp:].split()
   actf = a[actp:].split()

   actm = actf[0].strip("a=")
   actmo = float(actm)
   if actmo == 1.0:
      actcount += 1


   d = c[0].strip("V=")
   if len(cth[0]) > 6:
      dth = cth[0].strip("Vth=")
   else:
      dth = cth[1]
   j = i[0].strip("t=")
   if len(gef[0]) > 5:
      gem = gef[0].strip("G_E=")
   if len(gef[0]) < 5:
      gem = gef[1]
   if len(gibf[0]) > 6:
      gibm = gibf[0].strip("G_IB=")
   else:
      gibm = gibf[1]

   #if len(gibm) == 0:
   #   gibm = gibf[1]

   if len(gif[0]) > 5:
      gim = gif[0].strip("G_I=")
   else:
      gim = gif[1]
     
   v.append(d)
   vth.append(dth)
   t.append(j)
   ge.append(gem)
   gi.append(gim)
   gib.append(gibm)
   act.append(actm)



x = t
y = v
y2 = vth
y3 = ge
y4 = gi
y5 = gib


plt.figure(1)
plt.subplot(211)
plt.plot(x, y)
plt.plot(x, y2, 'r')
plt.ylabel('Membrane Rate')
plt.title('Membrane Activity')
plt.grid(True)


plt.subplot(212)
plt.ylabel('Conductance Rate')
plt.xlabel('Time (ms)\nG_E = green, G_I = yellow, G_IB = black\n Vth = red, V = blue')
plt.title('Conductance Activity           Fired  %.2f times' %(actcount)) 
if len(y3) > 0:
   plt.plot(x, y3, 'g')
if len(y4) > 0:
   plt.plot(x, y4, 'y')
if len(y5) > 0:
   plt.plot(x, y5, 'k')
plt.grid(True)
#plt.annotate(", xy = (30, 2.75), xytext='data')
plt.show()

sys.exit()
