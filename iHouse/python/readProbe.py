#!/usr/bin/env python
from matplotlib.pyplot import plot, legend, show, bar, figure, xticks, tight_layout
from numpy import array, dot, arange, mean, polyfit, ndarray, std, zeros
from collections import OrderedDict

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

filename = "/Users/slundquist/Desktop/ptLIF.txt"
#filename = "/Users/slundquist/Desktop/retONtoLif.txt"
#filename = "/Users/dpaiton/Documents/Work/LANL/workspace/iHouse/checkpoints/Checkpoint3000000/retONtoLif.txt"

#Values for range of frames
all_lines = True#All values if True
startTime = 2900000
#End must be under number of lines in file
endTime   = 3000000

numTCBins   = 2  #number of bins for time course plot
numHistBins = -1 #number of bins for histogram of weights (-1 means no histogram)

#Data structure for scale, and data array to store all the data
data = OrderedDict()
#Made time for data
#TIME MUST EXIST AND BE FIRST IN THIS LIST
data['t']                     = []

data['V']                    = []
data['Vth']                  = []
data['a']                    = []

##data['weights*']             = []
#data['prOjaTr15']             = []
#data['prOjaTr*']              = []
#data['prStdpTr*']             = []
#data['poIntTr']               = []

print "readProbe: Reading file..."
f = open(filename, 'r')
if (all_lines):
    lines = f.readlines()
else:
    firstLine = f.readline()
    firstLineSplit = splitLine(firstLine) #list of tuples. list[0] is always time. tuple is ('label','val')
    fileStartTime = float(firstLineSplit[0][1])
    if startTime < fileStartTime: #can't start from a time earlier than the first time
        startTime = fileStartTime
    if startTime != fileStartTime: #start time is somewhere past the beginning of the file
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
            line = f.readline()
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
linesSp = [splitLine(line) for line in lines]

print "readProbe: Parsing Keys..."
doHist = 0
#Loop through all keys given by user
for key in data.keys():
    print "readProbe: -Formatting key: '" + key + "'"
    #Get key value, without the * if it is there
    if key[len(key)-1] == "*":
        tok = key[:len(key) - 1]
    else:
        tok = key
    #Grab the value if element is the same as token
    if key[len(key)-1] == "*":
        if key == "weight*" and numHistBins != -1:
            doHist = 1
        else:
            doHist = 0
        print "readProbe: --Key occurs in multiple instances per line, computing bin edge values..."
        #Get all instance values in all time steps for given key
        #allVals is list of lists - [time][vals]
        #  num vals in each time step should equal the pre patch size
        allVals = [[float(x[1]) for x in lineSp if x[0][:min(len(x[0]), len(tok))] == tok] for lineSp in linesSp]
        minVal = min(min(allVals)) #max of all vals across all time
        maxVal = max(max(allVals)) #min of all vals across all time
        step = (maxVal - minVal) / float(numTCBins)
        if doHist:
            stepHist = (maxVal - minVal) / float(numHistBins)
            print "readProbe: --Binning the values (may take some time because of Histogam computations)..."
        else:
            print "readProbe: --Binning the values..."
        #Grab boundary points based on range and bins
        if step == 0:
            print "readProbe: --" + key + " Min and Max equivelant, defaulting to one bin."
            #data refers to list of all values in time. mean() will reduce number of instances to 1
            data[key] = [mean(x) for x in allVals]
            if doHist:
                numHistBins = 1
        else:
            boundList = list(arange(minVal, maxVal, step))
            boundList.append(maxVal)
            #Split data into another array based on bound list for each val in timestep
            tempVals = [[[a for a in val if a > boundList[i] and a <= boundList[i+1]]for i in range(len(boundList) - 1)] for val in allVals]
            if doHist:
                print "readProbe: ---Computing histogram information..."
                boundListHist = list(arange(minVal, maxVal, stepHist))
                boundListHist.append(maxVal)
                tempValsHist = [[len([a for a in val if a > boundListHist[i] and a <= boundListHist[i+1]]) for i in range(len(boundListHist) - 1)] for val in allVals]
                counts = zeros(numHistBins)
                ##TODO: make list comprehension
                for time in range(len(tempValsHist)):
                    for iBin in range(numHistBins):
                        counts[iBin] += tempValsHist[time][iBin]

            print "readProbe: --Computing line of best fit..."
            #Find best line of fit
            xVals = {}
            yVals = {}
            #Allocate arrays for dictionaries
            for i in range(numTCBins):
                xVals[i] = []
                yVals[i] = []

            #Iterate through everything to get data points for line of best fit
            for time in range(len(tempVals)):
                for i, bins in enumerate(tempVals[time]):
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
        data[key] = [float(x[1]) for lineSp in linesSp for x in lineSp if len(x[0]) == len(tok) and x[0][:len(tok)] == tok]
    print "readProbe: -Done formatting '"+key+"'."

print "readProbe: Done parsing keys."
print "readProbe: Creating time course plot..."
figure(0)
time = array(data['t'])
for key in data.keys():
    if key == 't':
        continue
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
        plot(time, plotMe, label=key)
legend()#bbox_to_anchor=(0., 1.02, 1., .102), ncol = 2, mode="expand", borderaxespad=0.,loc=3)
tight_layout()

if doHist:
   print "readProbe: Creating histogram plot..."
   fig1 = figure(1)
   maxWeight = max(max(max(tempVals))) #TODO: is there a better way to do this?
   xVals = arange(0,maxWeight,maxWeight/len(counts))
   bar(range(len(counts)),counts)
   xticks(arange(20),xVals,rotation='vertical')
   tight_layout()


print "readProbe: Script complete."

show()

