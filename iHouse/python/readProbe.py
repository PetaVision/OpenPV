#!/usr/bin/env python
from matplotlib.pyplot import plot, legend, show
from numpy import array, dot, arange, mean, polyfit, ndarray, std
from collections import OrderedDict

#filename = "/Users/slundquist/Desktop/LCALIF_31_31_0.txt"
filename = "/Users/slundquist/Desktop/retONtoLif.txt"

#Values for range of frames
all = False; #All values
startVal = 0
#End must be under number of lines in file
endVal = 100

numBins = 2

#Data structure for scale, and data array to store all the data
data = OrderedDict()
#Made time for data
#TIME MUST EXIST AND BE FIRST IN THIS LIST
data['t']                     = []

#data['V']                    = []
#data['Vth']                  = []
#data['Vadpt']                = []
#data['a']                    = []
#data['integratedSpikeCount'] = []

#data['weights*']             = []
data['prOjaTr15']             = []
data['prOjaTr*']              = []
data['prStdpTr*']             = []
data['poIntTr']               = []


def splitLine(line):
   #Split line by =
   lineSp = line.split("=")
   #Split further by spaces
   lineSp = [x.split() for x in lineSp]
   #Combine into one list
   lineSp = [a for x in lineSp for a in x]
   #Remove first element, which is probe name
   lineSp = lineSp[1:]
   #Group elements into tuples
   lineSp = zip(*[lineSp[i::2] for i in range(2)])
   return lineSp


if startVal < 0 and not all :
   print "Start value must be above 0"
#if endVal >= len(lines) and not all:
#   print "End value must be below the total time"

f = open(filename, 'r')
if (all):
   lines = f.readlines()
else:
   timeRange = range(startVal, endVal)
   lines = [f.readline() for i in timeRange]
f.close()

bounds = {}
stds = {}

linesSp = [splitLine(line) for line in lines]

for key in data.keys():
   if key[len(key)-1] == "*":
      tok = key[:len(key) - 1]
   else:
      tok = key
   #Grab the value if element is the same as token
   if key[len(key)-1] == "*":
      allVals = [[float(x[1]) for x in lineSp if x[0][:min(len(x[0]), len(tok))] == tok] for lineSp in linesSp]
      minVal = min(min(allVals))
      maxVal = max(max(allVals))
      #Grab boundary points based on bins
      step = (maxVal - minVal) / float(numBins)
      if step == 0:
         print key + " Min and Max are same value, defaulting to one bin"
         #Put in all one list, should be only one element in list
         data[key] = [mean(x) for x in allVals]
      else:
         boundList = list(arange(minVal, maxVal, step))
         boundList.append(maxVal)
         #Split data into another array based on bound list for each val in timestep
         tempVals = [[[a for a in val if a > boundList[i] and a <= boundList[i+1]]for i in range(len(boundList) - 1)] for val in allVals]
         #Find best line of fit
         xVals = {}
         yVals = {}
         #Allocate arrays for dictionaries
         for i in range(numBins):
            xVals[i] = []
            yVals[i] = []

         #Iterate through everything to get data points for line of best fit
         for time in range(len(tempVals)):
            for i, bins in enumerate(tempVals[time]):
               if len(bins) != 0:
                  yVals[i].extend(bins)
                  xVals[i].extend([data['t'][time] for i in range(len(bins))])
         #Find line of best fit, one for each bin
         #Output is polynomial value, as such: (slope, yintercept)
         data[key] = [polyfit(xVals[i], yVals[i], 1) if len(yVals[i]) != 0 else array([]) for i in range(numBins)]
         #Calculate standard deviation
         stds[key] = [std(yVals[i]) if len(yVals[i]) != 0 else array([]) for i in range(numBins)]
         bounds[key] = boundList
   else:
      data[key] = [float(x[1]) for lineSp in linesSp for x in lineSp if len(x[0]) == len(tok) and x[0][:len(tok)] == tok]

   #Save bounds

#for t in timeRange:
#   line = lines[t]
#   for key in data.keys():
#      tok = key + '='
#      listSp = line.rsplit(tok)
#      strVal = listSp[1]
#      ind = strVal[1:].find(' ');
#      if ind == -1:
#         data[key].append(float(strVal[:len(strVal) - 1]))
#      else:
#         data[key].append(float(strVal[:ind + 1]))

time = array(data['t'])
for key in data.keys():
   if key == 't':
      continue
   if key[len(key)-1] == "*":
      if type(data[key][1]) is ndarray:
         for i in range(numBins):
            if(len(data[key][i]) != 0):
               plotMe = dot(time, data[key][i][0]) + data[key][i][1]
               plot(time, plotMe, label=key + ' bin:(' + str(bounds[key][i]) + ',' + str(bounds[key][i+1]) + ')' + 'std:' + str(stds[key][i]))
      else:
         plotMe = array(data[key])
         plot(time, plotMe, label=key)
   else:
      plotMe = array(data[key])
      plot(time, plotMe, label=key)

legend(bbox_to_anchor=(0., 1.02, 1., .102), ncol = 2, mode="expand", borderaxespad=0.,loc=3)
show()
