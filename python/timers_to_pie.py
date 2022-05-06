#! /usr/bin/env python

#File to make pie charts off of timers
#Written by Sheng Lundquist
#Updated by Pete Schultz

import matplotlib.pyplot as plt
import pdb
import numpy as np
import sys
import os.path

if (len(sys.argv) == 1):
    print("timerPie.py path/to/timers.txt")
    print("Creates PNG pie charts of the times in the specified timers.txt file.")
    print("buildrunTimes.png shows the time spent building, running, and checkpointing.")
    print("objectTimes.png shows the time spent in each timer type.")
    print("timerTimes.png shows the time spent in each timer type.")
    print("These files are written to the same directory the timers.txt file is in.")
    exit(0)

timersFile = sys.argv[1]
outputDirectory = os.path.dirname(timersFile)
if len(outputDirectory) == 0:
    outputDirectory = '.'

buildruntime = float('nan')
buildtime = float('nan')
runtime = float('nan')
chkpointtime = float('nan')
overheadtime = float('nan')

objectNames = []
timerTypes  = []
timerTimes  = []

f = open(timersFile, 'r')
for l in f:
    if ': processor cycle time == ' not in l:
        continue
    if len(l) > 10 and l[:10] == 'StatsProbe':
        continue

    #Split via colon
    split = l.split(":")
    #Grab first split, which contains the name of the object
    objectName = split[0].strip()
    #Next split has the form " total time in [objtype] [timertype]"
    objectType = split[1].split()[-2]
    timerType  = split[1].split()[-1]
    #Grab last split, which has the form " processor cycle time == [time in ms]"
    timerTime = float(split[2].split()[-1])

    if timerType == 'buildrun':
        buildruntime = timerTime
        continue

    if timerType == 'build':
        buildtime = timerTime
        continue

    if timerType == 'run':
        runtime = timerTime
        continue

    if timerType == 'checkpoint':
        chkpointtime = timerTime
        continue

    if 'gpurecvsyn' in timerType:
        continue

    if 'gpuupdate' in timerType:
        continue

    if 'init' in timerType:
        continue

    objectNames.append(objectName)
    timerTypes.append(objectType + ' ' + timerType)
    timerTimes.append(timerTime)

f.close()

if np.isnan(buildruntime) or np.isnan(buildtime) or np.isnan(runtime) or np.isnan(chkpointtime):
    print("build+run timers were not found; skipping buildrun pie chart.", file=sys.stderr)
    exit(1)
else:
    overheadtime = buildruntime - buildtime - runtime - chkpointtime
    f = plt.figure()
    buildRunTimes = {}
    buildRunTimes['initializing'] = buildtime
    buildRunTimes['runloop'] = runtime
    buildRunTimes['checkpointing'] = chkpointtime
    buildRunTimes['untimed overhead'] = overheadtime

    pieceThreshold = 0.02 * buildruntime

    otherTimes = 0.0
    significantBuildRunTimes = {}
    for tname in buildRunTimes:
        ttime = buildRunTimes[tname]
        if ttime < pieceThreshold:
            otherTimes += buildRunTimes[tname]
        else:
            significantBuildRunTimes[tname] = ttime

    significantBuildRunTimes["Other"] = otherTimes

    sortedBuildRunTimes = dict(sorted(significantBuildRunTimes.items(), key=lambda p:p[1], reverse=True))

    plt.title('Overall Times')
    plt.pie(sortedBuildRunTimes.values(), labels = sortedBuildRunTimes.keys())
    plt.savefig(outputDirectory + "/buildrunTimes.png")

sumTimerTimes = sum(timerTimes)
runoverhead = runtime - sumTimerTimes

#Make 2 pies, one per layerName, one per timerName
timesByObject = {}
timesByType = {}
for objname, ttype, ttime in zip(objectNames, timerTypes, timerTimes):
    if objname in timesByObject:
        timesByObject[objname] += ttime
    else:
        timesByObject[objname] = ttime

    if ttype in timesByType:
        timesByType[ttype] += ttime
    else:
        timesByType[ttype] = ttime

timesByObject["Untimed Overhead"] = runoverhead
timesByType["untimed overhead"] = runoverhead

pieceThreshold = 0.02 * runtime;

otherObjectTimes = 0.0
significantTimesByObject = {}
for objname in timesByObject:
    objTime = timesByObject[objname]
    if objTime < pieceThreshold:
        otherObjectTimes += timesByObject[objname]
    else:
        significantTimesByObject[objname] = objTime

sortedTimesByObject = dict(sorted(significantTimesByObject.items(), key=lambda p:p[1], reverse=True))
sortedTimesByObject["Other"] = otherObjectTimes

otherTypeTimes = 0.0
significantTimesByType = {}
for typename in timesByType:
    typeTime = timesByType[typename]
    if typeTime < pieceThreshold:
        otherTypeTimes += timesByType[typename]
    else:
        significantTimesByType[typename] = typeTime

sortedTimesByType = dict(sorted(significantTimesByType.items(), key=lambda p:p[1], reverse=True))
sortedTimesByType["Other"] = otherTypeTimes

f = plt.figure()
plt.title('Run Time by Object')
plt.pie(sortedTimesByObject.values(), labels=sortedTimesByObject.keys())
plt.savefig(outputDirectory + "/objectTimes.png")

f = plt.figure()
plt.title('Run Time by Action Type')
plt.pie(sortedTimesByType.values(), labels=sortedTimesByType.keys())
plt.savefig(outputDirectory + "/timerTimes.png")
