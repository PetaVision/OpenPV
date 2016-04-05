#File to make pie charts off of timers
#TODO achieve this via reading in log file as opposed to specilized file
#Written by Sheng Lundquist

import matplotlib.pyplot as plt
import pdb
import numpy as np

timersFile = "/home/sheng/mountData/KITTI/KITTI_Deep_2X3frames/benchmark0/output.txt"
outputDirectory = "/home/sheng/mountData/KITTI/KITTI_Deep_2X3frames/benchmark0/"

f = open(timersFile, 'r')
fileLines = f.readlines()
f.close()

#Remove first 4 lines
fileLines.pop(0)
fileLines.pop(0)
fileLines.pop(0)

layerName = []
timerName = []
times = []

##Take out cumulation times
#totalTime = times.pop(0)
#labels.pop(0)

for l in fileLines:
    #Split via colon
    split = l.split(":")
    #Grab first split, which contains the name of the timer
    layerName.append(split[0].strip())
    timerName.append(split[1].split()[-2]+split[1].split()[-1])
    #Grab lasts split, which contains the time
    times.append(float(split[2].split()[-1]))

layerTimes = {}
timerTimes = {}
#Make 2 pies, one per layerName, one per timerName
for lname, tname, time in zip(layerName, timerName, times):
    if("column" in lname):
        continue
    #Remove column times
    if lname in layerTimes:
        layerTimes[lname] += time
    else:
        layerTimes[lname] = time

    if tname in timerTimes:
        timerTimes[tname] += time
    else:
        timerTimes[tname] = time


f = plt.figure()
plt.pie(layerTimes.values(), labels=layerTimes.keys())
plt.savefig(outputDirectory + "/layerTimes.png")

f = plt.figure()
plt.pie(timerTimes.values(), labels=timerTimes.keys())
plt.savefig(outputDirectory + "/timerTimes.png")


#plt.pie(times, labels=labels)
#plt.show()
#pdb.set_trace()
