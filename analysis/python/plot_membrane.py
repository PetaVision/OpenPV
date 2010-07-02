import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import PVReadWeights as rw
import PVConversions as conv


if len(sys.argv) < 2:
   print "usage: membrane filename l1-activity"
   print len(sys.argv)
   sys.exit()



checkarg = sys.argv[1]

check = checkarg.find("Inh")
acheck = checkarg.find("image")
rocheck = checkarg.find("retinaOn")
l1check = checkarg.find("l1")

print check
print acheck
print rocheck
print l1check


a = open(sys.argv[1], "r")
v   = []
t   = []
t2  = []
vth = []
ge  = []
gi  = []
gib = []
aa  = []
aa2 = []

if check <= 0 and acheck == -1 and rocheck == -1 and l1check == -1:
   for line in a:
      a = line
      b = a.find("V=")
      h = a.find("t=")
      bth = a.find("Vth=")
      gep = a.find("G_E=")
      gip = a.find("G_I=")
   
   
      c = a[b:].split()
      cth = a[bth:].split()
      i = a[h:].split()
      gef = a[gep:].split()
      gif = a[gip:].split()

      d = c[0].strip("V=")
      dth = cth[0].strip("Vth=")
      j = i[0].strip("t=")
      gem = gef[1]
      gim = gif[1]

      v.append(d)
      vth.append(dth)
      t.append(j)
      ge.append(gem)
      gi.append(gim)


if check >= 0 and acheck == -1 and rocheck == -1 and l1check == -1:
   for line in a:
      a = line
      b = a.find("V=")
      h = a.find("t=")
      bth = a.find("Vth=")
      gep = a.find("G_E=")
      gibp = a.find("G_IB=")   
   
      c = a[b:].split()
      cth = a[bth:].split()
      i = a[h:].split()
      gef = a[gep:].split()
      gibf = a[gibp:].split()

      d = c[0].strip("V=")
      dth = cth[0].strip("Vth=")
      j = i[0].strip("t=")
      if len(gef[0]) > 5:
         gem = gef[0].strip("G_E=")
      if len(gef[0]) < 5:
         gem = gef[1]
      if len(gef[0]) > 5:
         gibm = gibf[0].strip("G_IB=")
      if len(gef[0]) < 5:
         gibm = gibf[1]
      if len(gibm) == 0:
         gibm = gibf[1]
     
      v.append(d)
      vth.append(dth)
      t.append(j)
      ge.append(gem)
      gib.append(gibm)

if acheck == 0:
   for line in a:
      a = line
      b = a.find("a=")
      c = a[b:].split()
      d = c[0].strip("a=")
      
      h = a.find("t=")
      i = a[h:].split()
      j = i[0].strip("t=")
      t.append(j)
      aa.append(d)
   
   
   
   x = t
   y = aa
   #print aa
   
   plt.figure(1)
   plt.subplot(211)
   plt.plot(x, y, 'r')
   plt.ylabel('Membrane Rate')
   plt.title('Membrane Activity')
   plt.grid(True)
   plt.show()
   sys.exit()


if rocheck == 0:
   count = 0
   for line in a:
      a = line
      b = a.find("a=")
      c = a[b:].split()
      d = c[0].strip("a=") 
      do = float(d)

      if do == 1.0:
         count = count + 1

      h = a.find("t=")
      i = a[h:].split()
      j = i[0].strip("t=")

      t.append(j)
      aa.append(d)

   count2 = 0
   a2 = open(sys.argv[2], "r")
   for line in a2:
      a2 = line
      b2 = a2.find("a=")
      c2 = a2[b2:].split()
      d2 = c2[0].strip("a=") 
      do2 = float(d2)

      if do2 == 1.0:
         count2 = count2 + 1

      h2 = a2.find("t=")
      i2 = a2[h2:].split()
      j2 = i2[0].strip("t=")

      t2.append(j2)
      aa2.append(d2)
   
   
   
   
   print "On count = ",count
   x = t
   y = aa

   print "Off count = ", count2
   x2 = t2
   y2 = aa2
   #print aa
   
   plt.figure(1)
   plt.subplot(211)
   plt.plot(x, y, 'r')
   plt.ylabel('Membrane Rate')
   plt.title('Membrane Activity')
   plt.xlabel('Times On fired = %d' %(count))
   plt.grid(True)

   plt.subplot(212)
   plt.plot(x2, y2, 'r')
   plt.ylabel('Membrane Rate')
   plt.xlabel('Times Off fired = %d' %(count2))
   plt.grid(True)

   plt.show()
   sys.exit()



if l1check == 0:
   count = 0
   for line in a:
      a = line
      b = a.find("a=")
      c = a[b:].split()
      d = c[0].strip("a=") 
      do = float(d)

      if do == 1.0:
         count = count + 1

      h = a.find("t=")
      i = a[h:].split()
      j = i[0].strip("t=")

      t.append(j)
      aa.append(d)

   count2 = 0
   a2 = open(sys.argv[2], "r")
   for line in a2:
      a2 = line
      b2 = a2.find("a=")
      c2 = a2[b2:].split()
      d2 = c2[0].strip("a=") 
      do2 = float(d2)

      if do2 == 1.0:
         count2 = count2 + 1

      h2 = a2.find("t=")
      i2 = a2[h2:].split()
      j2 = i2[0].strip("t=")

      t2.append(j2)
      aa2.append(d2)
   
   
   
   
   print "L1 count = ",count
   x = t
   y = aa

   print "L1Inh count = ", count2
   x2 = t2
   y2 = aa2
   #print aa
   
   plt.figure(1)
   plt.subplot(211)
   plt.plot(x, y, 'r')
   plt.ylabel('Membrane Rate')
   plt.title('Membrane Activity')
   plt.xlabel('Times L1 fired = %d' %(count))
   plt.grid(True)

   plt.subplot(212)
   plt.plot(x2, y2, 'r')
   plt.ylabel('Membrane Rate')
   plt.xlabel('Times L1Inh fired = %d' %(count2))
   plt.grid(True)

   plt.show()
   sys.exit()




x = t
y = v
y2 = vth
y3 = ge
y4 = gi
y5 = gib

#print y5
#sys.exit()

plt.figure(1)
plt.subplot(211)
plt.plot(x, y)
plt.plot(x, y2, 'r')
plt.ylabel('Membrane Rate')
plt.title('Membrane Activity')
plt.grid(True)

if check <= 0:
   plt.subplot(212)
   plt.ylabel('Conductance Rate')
   plt.xlabel('Time (s)\nG_E = green, G_I = yellow')
   plt.title('Conductance Activity') 
   plt.plot(x, y3, 'g')
   plt.plot(x, y4, 'y')
   plt.grid(True)
   #plt.annotate(", xy = (30, 2.75), xytext='data')
   plt.show()

if check >= 0:
   plt.subplot(212)
   plt.ylabel('Conductance Rate')
   plt.xlabel('Time (s)\nG_E = green, G_IB = yellow')
   plt.title('Conductance Activity') 
   plt.plot(x, y3, 'g')
   plt.plot(x, y5, 'y')
   plt.grid(True)
   #plt.annotate(", xy = (30, 2.75), xytext='data')
   plt.show()
