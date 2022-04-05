#File to make pie charts off of timers
#TODO achieve this via reading in log file as opposed to specilized file
#Written by Sheng Lundquist

import matplotlib.pyplot as plt
import pdb
import numpy as np
import sys
import os.path

if (len(sys.argv) == 1):
    print("timerPie.py path/to/timers.txt")
    print("Creates PNG pie charts of the times in the specified timers.txt file.")
    print("layerTimes.png shows the time spent in each object.")
    print("timerTimes.png shows the time spent in each timer type.")
    print("These files are written to the same directory the timers.txt file is in.")
    exit(0)

timersFile = sys.argv[1]
outputDirectory = os.path.dirname(timersFile);
if len(outputDirectory) == 0:
    outputDirectory = '.'

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
    if len(l) > 10 and l[:10] == 'StatsProbe':
        continue
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
