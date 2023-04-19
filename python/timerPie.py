#! /usr/bin/env python

#File to make pie charts off of timers
#Written by Sheng Lundquist
#Updated by Pete Schultz

import matplotlib
import matplotlib.pyplot as plt
import pdb
import numpy as np
import sys
import os.path

pltdefault = plt.rcParams
basefontsize = plt.rcParams.get('font.size')

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

# For each pie chart, values less than this percentage of the total will be lumped into
# an "other" category
cutoffpct = 0.02

labeltimecutoffpct = 0.04
# For each pie chart, the execution time for each category will be labeled only if the
# percentage exceeds this value.

# Default for autopct is to label wedges as percent.
# The function time_seconds_string recovers the absolute value and
# expresses it as a string, for autopct to draw in the wedge
def time_seconds_string(pct, values):
    total = 0.001 * sum(values) # values will be in milliseconds; convert to seconds
    quantity = pct/100.0 * total
    if pct >= 100.0 * labeltimecutoffpct:
       label_string = '{:.0f} s'.format(quantity)
    else:
       label_string = ''
    print('pct = {:f}, total = {:f}, label_string = {:s}'.format(pct, total, label_string))
    return label_string

if np.isnan(buildruntime) or np.isnan(buildtime) or np.isnan(runtime) or np.isnan(chkpointtime):
    print("build+run timers were not found; skipping buildrun pie chart.", file=sys.stderr)
    exit(1)

overheadtime = buildruntime - buildtime - runtime - chkpointtime
buildRunTimes = {}
buildRunTimes['initializing'] = buildtime
buildRunTimes['runloop'] = runtime
buildRunTimes['checkpointing'] = chkpointtime
buildRunTimes['untimed overhead'] = overheadtime

pieceThreshold = cutoffpct * buildruntime

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

f = plt.figure()
plt.title('Overall Times', fontsize=basefontsize+6.0)
plt.pie(
    sortedBuildRunTimes.values(),
    labels = sortedBuildRunTimes.keys(),
    autopct = lambda pct: time_seconds_string(pct, sortedBuildRunTimes.values()),
    wedgeprops = {'edgecolor':'k'})
f.text(0.5, 0.02,
       'Total execution time {t:.0f} seconds'.format(t=0.001*buildruntime),
       horizontalalignment='center',
       fontsize=basefontsize+2.0)
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

pieceThreshold = cutoffpct * runtime;

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
plt.title('Run Time by Object', fontsize=basefontsize+6.0)
plt.pie(
    sortedTimesByObject.values(),
    labels=sortedTimesByObject.keys(),
    autopct = lambda pct: time_seconds_string(pct, sortedTimesByObject.values()),
    wedgeprops = {'edgecolor': 'k'})

f.text(0.5, 0.02,
       'Total runtime {t:.0f} seconds'.format(t=0.001*runtime),
       horizontalalignment='center',
       fontsize=basefontsize+2.0)

plt.savefig(outputDirectory + "/objectTimes.png")

f = plt.figure()
plt.title('Run Time by Action Type', fontsize=basefontsize+6.0)
plt.pie(
    sortedTimesByType.values(),
    labels=sortedTimesByType.keys(),
    autopct = lambda pct: time_seconds_string(pct, sortedTimesByType.values()),
    wedgeprops = {'edgecolor': 'k'})
f.text(0.5, 0.02,
       'Total runtime {t:.0f} seconds'.format(t=0.001*runtime),
       horizontalalignment='center',
       fontsize=basefontsize+2.0)
plt.savefig(outputDirectory + "/timerTimes.png")
