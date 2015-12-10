#!/usr/bin/env python
#####
## readProbe.py
##   Read in PetaVision probe output
##   Display time-course plots and histograms of values
##   Should work with any standard probe. Currently tested with LIF, OjaSTDPConn, LCALIF probes.
##
##Dylan Paiton and Sheng Lundquist
#####

import matplotlib
from numpy import array, dot, arange, mean, polyfit, ndarray, std, zeros, nonzero
from collections import OrderedDict

#*************************#
#          PARAMS         #
#*************************#
#Paths
#workspaceDir = "/Users/dpaiton/Documents/Work/Lanl/workspace"
#workspaceDir = "/home/dpaiton/workspace"
#filename     = "/Users/slundquist/Desktop/ptLIF.txt"
filename     = "/Users/slundquist/Desktop/lifLat_Center.txt"
figOutDir    = "/Users/slundquist/Desktop/figOut/"
figRootName  = 'onVer_weights'
#filename     = workspaceDir+"/iHouse/checkpoints/Checkpoint3010000/retONtoLifHor.txt"
#figOutDir    = workspaceDir+"/iHouse/checkpoints/Checkpoint3010000/analysis/probeFigs/"
#figRootName  = 'onVer_weights'

#Values for range of frames
all_lines = True    #All values if True
startTime = 1500000
endTime   = 1600000  #End must be under number of lines in file

#Other flags
numTCBins   = 2     #number of bins for time course plot
numHistBins = 20    #number of bins for histogram of weights (-1 means no histogram)
weightMap   = False  #if True, display a weight map
doLegend    = True#if True, time graph will have a legend
dispFigs    = False#if True, display figures. Otherwise, print them to file.

#Flags for weightMap
numWeights = 25 #number of weights in map

#Data structure for scale, and data array to store all the data
data = OrderedDict()
#Made time for data
#TIME MUST EXIST AND BE FIRST IN THIS LIST
data['t']                     = []

#data['V']                    = []
#data['Vth']                  = []
#data['a']                    = []

data['weight_0_*']               = []
#data['weight_0_0']               = []
#data['weight_0_1']               = []
#data['weight_0_2']               = []
#data['weight_0_3']               = []
#data['weight_0_4']               = []
#data['prOjaTr_0_0']               = []
#data['prOjaTr_0_1']               = []
#data['prOjaTr_0_2']               = []
#data['prOjaTr_0_3']               = []
#data['prOjaTr_0_4']               = []

#data['weight5']               = []
#data['weight6']               = []
#data['weight7']               = []
#data['weight8']               = []
#data['weight9']               = []
#data['weight10']              = []
#data['weight11']              = []
#data['weight12']              = []
#data['weight13']              = []
#data['weight14']              = []
#data['weight15']              = []
#data['weight16']              = []
#data['weight17']              = []
#data['weight18']              = []
#data['weight19']              = []
#data['weight20']              = []
#data['weight21']              = []
#data['weight22']              = []
#data['weight23']              = []
#data['weight24']              = []
#data['weight*']               = []
#data['prOjaTr*']              = []
#data['prStdpTr*']             = []
#data['poOjaTr']               = []
#data['poStdpTr']              = []
#data['poIntTr']               = []
#data['ampLTD']                = []

#*************************#
#        MAIN CODE        #
#*************************#

def splitLine(line):
   #Split line by :
   lineSp = line.split(":")
   assert(len(lineSp) == 2) ##only one colon in params name
   lineSp = lineSp[1]
   #Split line by =
   lineSp = lineSp.split("=")
   #Split further by spaces
   lineSp = [x.split() for x in lineSp]
   #Combine into one list
   lineSp = [a for x in lineSp for a in x]
   #Group elements into tuples
   lineSp = zip(*[lineSp[i::2] for i in range(2)])
   return lineSp

#Must be done before importing pyplot (or anything from pyplot)
if not dispFigs:
    matplotlib.use('Agg')
from matplotlib.pyplot import plot, legend, show, bar, figure, xticks, tight_layout, gray

#Imports for weightMap
if weightMap:
    from numpy import mean, sqrt, reshape
    from matplotlib.pyplot import imshow, grid

if numTCBins <= 0:
    numTCBins = 1
    print "readProbe: WARNING: numTCBins <= 0, which is not allowed. Setting numTCBins to 1."
if numHistBins == 0:
    numHistBins = -1
    print "readProbe: WARNING: numHistBins == 0, which is not allowed. Setting numHistBins to -1."

print "readProbe: Reading file..."
f = open(filename, 'r')
if (all_lines):
    lines = f.readlines()
else:
    firstLine = f.readline()
    firstLineSplit = splitLine(firstLine) #list of tuples. list[0] is always time. tuple is ('label','val')
    fileStartTime = float(firstLineSplit[0][1])
    assert endTime > fileStartTime, "readProbe: endTime ("+str(endTime)+") is <= fileStartTime ("+str(fileStartTime)+")"
    if startTime < fileStartTime: #can't start from a time earlier than the first time
        startTime = fileStartTime
        print "readProbe: WARNING: startTime is less than the file's start time. Setting startTime = fileStartTime"
    assert endTime > startTime, "readProbe: endTime must be greater than startTime."
    timeOffset = startTime - fileStartTime #now we know how many lines forward we need to go
    lineLength = len(firstLine)
    f.seek(lineLength*timeOffset,0)
    line = f.readline() #this possibly(probably?) does not start at the beginning of the line
    line = f.readline() #is definitely a full line
    currentTime = float(splitLine(line)[0][1])
    while currentTime != startTime:
        if currentTime > startTime: #shouldn't have to go very far back, assuming that the length of the first line approximately matches the average line length
            tempTime = currentTime #to make sure we actually back up
            numBack = 3 #should allow you to back up a single line.
            while tempTime >= currentTime:
                if (-len(line)*numBack < 0):
                    f.seek(0,0)
                    line = f.readline() #might not be full line
                else:
                    f.seek(-len(line)*numBack,1) #back up an extra line because you have to jump forward to be sure you have a whole line
                    line = f.readline() #might not be full line
                    line = f.readline()
                tempTime = float(splitLine(line)[0][1])
                numBack += 1
            currentTime = tempTime
        else: #currentTime < startTime
            line = f.readline()
            currentTime = float(splitLine(line)[0][1])
    #Should be at the right spot in the file now
    lines = []
    lines.append(line) #put the first line where it belongs
    while True:
        if currentTime < endTime:
            line = f.readline()
        else:
            break
        if len(line) == 0: #make sure endTime is not too far
            if endTime > currentTime:
                print "readProbe: WARNING: Your endTime is greater than the max time in the file. Stopping at time "+str(currentTime)
            break
        currentTime += 1
        lines.append(line)
f.close()

print "readProbe: Formatting file into data structure..."
bounds = {}
stds = {}
lines = [splitLine(line) for line in lines]

if weightMap:
    #TODO: assert that the sqrt of numWeights exists
    wMap = zeros(numWeights)
    keyIndex = 0

print "readProbe: Parsing Keys..."
doHist = 0
#Loop through all keys given by user
for key in data.keys():
    workingLines = lines[:]
    print "readProbe: -Formatting key: '" + key + "'"
    #Get key value, without the * if it is there
    if key[len(key)-1] == "*":
        tok = key[:len(key) - 1]
    else:
        tok = key
    #Grab the value if element is the same as token
    if key[len(key)-1] == "*":
        if key == "weight_0_*" and numHistBins is not -1:
            doHist = 1
        else:
            doHist = 0
        print "readProbe: --Key occurs in multiple instances per line, computing bin edge values..."
        #Get all instance values in all time steps for given key
        #workingLines is now a list of lists - [time][vals]
        #  num vals in each time step should equal the pre patch size
        workingLines = [[float(x[1]) for x in lineSp if x[0][:min(len(x[0]), len(tok))] == tok] for lineSp in workingLines]
        if doHist:
            allVals = [[float(x[1]) for x in lineSp if x[0][:min(len(x[0]), len(tok))] == tok] for lineSp in lines]
        minVal = min(min(workingLines)) #max of all vals across all time
        maxVal = max(max(workingLines)) #min of all vals across all time
        if doHist:
            maxWeight = maxVal
        step = (maxVal - minVal) / float(numTCBins)
        if doHist:
            stepHist = (maxVal - minVal) / float(numHistBins)
            print "readProbe: --Binning the values (may take some time because of Histogam computations)..."
        else:
            print "readProbe: --Binning the values..."
        #Grab boundary points based on range and bins
        if step == 0 or numTCBins==1: #TODO: This is kind of funky... can only hist if numTCBins > 1
            #data refers to list of all values in time. mean() will reduce number of instances to 1
            data[key] = [mean(x) for x in workingLines]
            if step == 0:
                print "readProbe: --" + key + " Min and Max equivelant, defaulting to one bin."
        else:
            boundList = list(arange(minVal, maxVal, step))
            boundList.append(maxVal)
            #Split data into another array based on bound list for each val in timestep
            workingLines = [[[a for a in val if a > boundList[i] and a <= boundList[i+1]]for i in range(len(boundList) - 1)] for val in workingLines]
            if doHist:
                print "readProbe: ---Computing histogram information..."
                boundListHist = list(arange(minVal, maxVal, stepHist))
                boundListHist.append(maxVal)
                tempValsHist = [[len([a for a in val if a > boundListHist[i] and a <= boundListHist[i+1]]) for i in range(len(boundListHist) - 1)] for val in allVals]
                del allVals #will not free until garbage collection
                counts = zeros(numHistBins)
                ##TODO: make list comprehension and negate need for tempValsHist list
                for time in range(len(tempValsHist)):
                    for iBin in range(numHistBins):
                        counts[iBin] += tempValsHist[time][iBin]
                del tempValsHist #will not free until garbage collection

            print "readProbe: --Computing line of best fit..."
            #Find best line of fit
            xVals = {}
            yVals = {}
            #Allocate arrays for dictionaries
            for i in range(numTCBins):
                xVals[i] = []
                yVals[i] = []

            #Iterate through everything to get data points for line of best fit
            for time in range(len(workingLines)):
                for i, bins in enumerate(workingLines[time]):
                    if len(bins) != 0:
                        yVals[i].extend(bins)
                        xVals[i].extend([data['t'][time] for i in range(len(bins))])
            #Find line of best fit, one for each bin
            print "readProbe: --Formatting data and computing the standard deviation..."
            #Output (data) is polynomial value, as such: (slope, yintercept)
            data[key] = [polyfit(xVals[i], yVals[i], 1) if len(yVals[i]) != 0 else array([]) for i in range(numTCBins)]
            #Calculate standard deviation
            stds[key] = [std(yVals[i]) if len(yVals[i]) != 0 else array([]) for i in range(numTCBins)]
            bounds[key] = boundList
    else:
        data[key] = [float(x[1]) for lineSp in workingLines for x in lineSp if len(x[0]) == len(tok) and x[0][:len(tok)] == tok]

    #Calculate weight map if requested
    if weightMap:
        if 'weight' in key and '*' not in key:
            print "readProbe: --Computing weight map..."
            wMap[keyIndex] = mean(data[key])
            keyIndex += 1
    print "readProbe: -Done formatting '"+key+"'"

if weightMap:
    #TODO: Make sure sqrt(numWeights) exists
    wMap = reshape(wMap,(sqrt(numWeights),sqrt(numWeights))) #reshape(vect, (nRows,nCols)), writes cols first

print "readProbe: Done parsing keys."
print "readProbe: Creating time course plot..."
fig0 = figure(0)
time = array(data['t'])
for key in data.keys():
    if key == 't':
        continue
    if key == 'a':
       print "Num activity: " + str(len(nonzero(data[key])[0]))
    if key[len(key)-1] == "*":
        if type(data[key][1]) is ndarray:
            for i in range(numTCBins):
                if(len(data[key][i]) != 0):
                    plotMe = time * data[key][i][0] + data[key][i][1]
                    plot(time, plotMe, label=key + ' bin:(' + str(bounds[key][i]) + ',' + str(bounds[key][i+1]) + ')' + 'std:' + str(stds[key][i]))
        else:
            plotMe = array(data[key])
            plot(time, plotMe, label=key)
    else:
        plotMe = array(data[key])
        if len(plotMe) != 0:
            plot(time, plotMe, label=key)
if doLegend:
    legend()#bbox_to_anchor=(0., 1.02, 1., .102), ncol = 2, mode="expand", borderaxespad=0.,loc=3)
tight_layout()

if doHist:
    print "readProbe: Creating histogram plot..."
    fig1 = figure(1)
    xVals = arange(0,maxWeight,maxWeight/len(counts))
    bar(range(len(counts)),counts)
    xticks(arange(20),xVals,rotation='vertical')
    tight_layout()

if weightMap:
    print "readProbe: Creating weight map..."
    fig2 = figure(2)
    imshow(wMap,aspect='auto',extent=[0,sqrt(numWeights),0,sqrt(numWeights)])
    grid(color='white')

if dispFigs:
    print "readProbe: Displaying figures..."
    show()
else:
    print "readProbe: Saving figure(s)..."
    from os import path, makedirs
    if not path.exists(figOutDir):
        makedirs(figOutDir)
    fig0.savefig(figOutDir+figRootName+"_timeCourse.png")
    if doHist:
        fig1.savefig(figOutDir+figRootName+"_hist.png")
    if weightMap:
        fig2.savefig(figOutDir+figRootName+"_weightMap.png")

print "readProbe: Script complete."
