import matplotlib.pyplot as plt
from numpy import array, dot
from collections import OrderedDict

filename = "/Users/slundquist/Desktop/LCALIF_31_31_0.txt"

#Values for range of frames
all = True; #All values
startVal = 4000
endVal = 6000

t = []
#Data that is to be pulled out from the probe
data = {
      'G_E':           [],
      'G_I':           [],
      'G_IB':          [],
      'dynVthRest':    [],
      'V':             [],
      'Vth':           [],
      'a':             []
}

#Which data points needs to be plotted
plotData = OrderedDict(
)
plotData['V'] = 1
plotData['Vth'] = 1
plotData['a'] = 10


f = open(filename, 'r')
lines = f.readlines()
f.close()

if startVal < 0:
   print "Start value must be above 0"
if endVal >= len(lines):
   print "End value must be below the total time"

if (all):
   timeRange = range(len(lines))
else:
   timeRange = range(startVal, endVal)

#Made time for data
data['t'] = [];

for t in timeRange:
   line = lines[t]
   for key in data.keys():
      tok = key + '='
      listSp = line.rsplit(tok)
      strVal = listSp[1]
      ind = strVal[1:].find(' ');
      if ind == -1:
         data[key].append(float(strVal[:len(strVal) - 1]))
      else:
         data[key].append(float(strVal[:ind + 1]))

time = array(data['t'])
for key in plotData.keys():
   plotMe = array(data[key])
   plotMe = dot(plotMe, plotData[key])
   plt.plot(time, plotMe, label=key)

plt.legend(loc=3)
plt.show()
