#!/usr/bin/env python
#####
## readProbe.py
##   Read in PetaVision probe output
##   Display time-course plots and histograms of values
##   Should work with any standard probe. Currently tested with LIF, OjaSTDPConn, LCALIF probes.
##
##Dylan Paiton and Sheng Lundquist
#####
import sys
import matplotlib

from readProbeFunc import *
from readProbeParams import *

#Must be done before importing pyplot (or anything from pyplot)
if not dispFigs:
    matplotlib.use('Agg')
from os import path, makedirs
from matplotlib.pyplot import *

#Error checking
if len([i for i in scale.keys() if i not in data.keys()]) > 0:
    print "readProbe: WARNING: Some of your key values for the scale dictionary do not match anything in the data dictionary. They will be ignored."
if weightMap:
    if 'weight*' not in data.keys():
        print "readProbe: WARNING: weight* is not set in the data dictionary, but weightMap flag is true. Setting weightMap to false."
        weightMap = False
if not path.exists(probeFileDir):
    sys.exit("readProbe: ERROR: probeFileDir ("+probeFileDir+") does not exist!")

#Main loop
for filenameTup in filenames:
    filename = filenameTup[1]

    if not dispFigs:
        figOutDir = rootFigOutDir+"/"+filenameTup[0]+"/"
        if not path.exists(figOutDir):
            makedirs(figOutDir)

    print "\n---------------"
    print "readProbe: Reading file "+filename
    lines = readProbeFile(filename,startTime,endTime) #lines is [time][char]

    print "readProbe: Formatting file into data structure..."
    lines = [splitLine(line) for line in lines] #lines is now [time][variable][(key),(val)]

    numTimeSteps  = len(lines) #Uniform for all keys
    numArbors     = {}
    numPreNeurons = {}
    numPreConns   = {}
    stds          = {}

    print "readProbe: Parsing Keys..."
    for key in data.keys():
        specificKey = True 
        if key[len(key)-1] == "*": #Get key value, without the * if it is there
            tok = key[:len(key) - 1]
            specificKey = False
        else:
            tok = key

        #Check to be sure that the tokens (keys) listed in data are actually in the probe's output
        checkTok = [[[tok in string for string in tup] for tup in line] for line in lines]
        if not any(checkTok[:]):
            sys.exit("readProbe: ERROR: Token '"+tok+"' was not found in the input file. Exiting program.")

        if key not in scale: # Set scale for plot to 1 if not defined
            scale[key] = 1

        workingLines = lines[:] #Make a single copy of the lines, use this copy throughout the loop

        numArbors[tok] = getNumArbors(tok,workingLines[0]) #Num arbors should be the same for each time step

        #Working lines is [time][preNeuron] and filtered for key of interest
        if specificKey:
            workingLines = [[float(part[1]) for part in line if part[0] == tok] for line in workingLines]
        else:
            workingLines = [[float(part[1]) for part in line if part[0].split('_')[0] == tok] for line in workingLines]

        #Total number of pre 
        numPreConns[tok] = len(workingLines[0])
        numPreNeurons[tok] = numPreConns[tok] / numArbors[tok]

        #workingLines is now a list of lists of lists- [arbor][preNeuron][time]
        #  number of preNeuron vals in each time step should equal the pre patch size
        workingLines = [[[workingLines[timeIndex][preIndex] for timeIndex in xrange(numTimeSteps)] for preIndex in range(numPreNeurons[tok]*arborID,numPreNeurons[tok]*arborID+numPreNeurons[tok])] for arborID in xrange(numArbors[tok])]

        print "readProbe: -Formatting key: '" + key + "'"
        if key[len(key)-1] == "*": #User has asked for all elements of a particular name
            if weightMap and tok == 'weight':
                print "readProbe: --Creating weight map..."
                wMap = [[mean(workingLines[arborID][neuronID][:]) for neuronID in range(numPreNeurons[tok])] for arborID in range(numArbors[tok])] #List of maps, one for each arbor
                if sqrt(numPreNeurons[tok])%1 > 0:
                    print "readProbe: WARNING: numPreNeurons["+tok+"] is not a perfect square! Using ceil() to avoid overflow."
                    squareVal = ceil(sqrt(numPreNeurons[tok]))
                else:
                    squareVal = sqrt(numPreNeurons[tok])

                wMap = [reshape(array(wMap[arborID]),(sqrt(numPreNeurons[tok]),sqrt(numPreNeurons[tok]))) for arborID in range(numArbors[tok])] #reshape(vect, (nRows,nCols)), writes cols first
                
            if timePlot:
                print "readProbe: --Binning values for the time plot..."
                stds[key] = []

                #get min/max/step across all neurons of the same name and all time
                minVals    = [min(min(workingLines[arborID][:][:])) for arborID in range(numArbors[tok])]
                maxVals    = [max(max(workingLines[arborID][:][:])) for arborID in range(numArbors[tok])]
                stepWidths = [(maxVals[arborID] - minVals[arborID]) / float(numTCBins) for arborID in range(numArbors[tok])]

                boundList = [list(arange(minVals[arborID], maxVals[arborID], stepWidths[arborID])) if stepWidths[arborID] != 0.0 else [0] for arborID in range(numArbors[tok])] # List of separators (edges) for bins
                for arborID in range(numArbors[tok]): #TODO: make inline?
                    boundList[arborID].append(maxVals[arborID]+1) #must be bigger so everything fits into the bin

                #workingLines is now [arborID][time][preNeuron]
                workingLines = [[[workingLines[arborID][preNeuron][timeStep]
                    for preNeuron in range(numPreNeurons[tok])] 
                    for timeStep in range(numTimeSteps)]
                    for arborID in range(numArbors[tok])]
                #workingLines is now [arborID][time][bin][preNeuron]
                workingLines = [[[[preNeuronVal 
                    for preNeuronVal in timeVals if preNeuronVal >= boundList[arborID][boundEdge] and preNeuronVal < boundList[arborID][boundEdge+1]]
                    for boundEdge in range(len(boundList[arborID])-1)] 
                    for timeVals in workingLines[arborID]] 
                    for arborID in range(numArbors[tok])]

                print "readProbe: --Computing line of best fit..."
                for arborID in range(numArbors[tok]):
                    #Find best line of fit
                    xVals = {}
                    yVals = {}
                    #Allocate arrays for dictionaries
                    for binKey in range(numTCBins):
                        xVals[binKey] = []
                        yVals[binKey] = []
                    #Iterate through everything to get data points for line of best fit
                    for time in range(numTimeSteps):
                        for binNo, bins in enumerate(workingLines[arborID][time]):
                            if len(bins) != 0:
                                yVals[binNo].extend(bins)
                                xVals[binNo].extend([data['t'][0][0][time] for binNo in range(len(bins))]) #Time always has 1 arbor (index 0) and 1 pre-neuron (index 0)
                    data[key].append([polyfit(xVals[binNo], yVals[binNo], 1) if len(yVals[binNo]) != 0 else array([]) for binNo in range(numTCBins)])
                    #Calculate standard deviation
                    stds[key].append([std(yVals[binNo]) if len(yVals[binNo]) != 0 else array([]) for binNo in range(numTCBins)])
        else:
            data[key] = workingLines
        print "readProbe: -Done formatting key '"+key+"'"

    time = array(data['t'][0][0][:]) #data[key][arbor][preNeuron]

    if weightMap:
        tok = 'weight'
        for arborID in range(numArbors[tok]):
            figure()
            imshow(wMap[arborID],aspect='auto',extent=[0,sqrt(numPreNeurons[tok]),0,sqrt(numPreNeurons[tok])])
            grid(color='white')
            colorbar()
            if not dispFigs:
                savefig(figOutDir+rootFigName+"_"+filenameTup[0]+"_weightMap"+str(arborID)+".png")
                clf()

    if timePlot:
        for key in data.keys():
            if key == 't':
                continue

            if key[len(key)-1] == "*": #Get key value, without the * if it is there
                tok = key[:len(key) - 1]
            else:
                tok = key

            if key[len(key)-1] == "*":
                for arborID in range(numArbors[tok]):
                    figure()
                    for TCBin in range(numTCBins):
                        if(len(data[key][arborID][TCBin]) != 0):
                            plotMe = time * data[key][arborID][TCBin][0] + data[key][arborID][TCBin][1]
                            plot(time, scale[key]*plotMe, label=key+'_a'+str(arborID)+' std:'+str(stds[key][arborID][TCBin]))
                    if doLegend:
                        legend()#bbox_to_anchor=(0., 1.02, 1., .102), ncol = 2, mode="expand", borderaxespad=0.,loc=3)
                    tight_layout()
                    if not dispFigs:
                        savefig(figOutDir+rootFigName+"_"+filenameTup[0]+"_timeCourseAvg"+str(arborID)+".png")
                        clf()

        didPlot = False #Only true if plot is created below
        figure()
        for key in data.keys(): #must repeat loop because we want all of these plots to be on one figure
            if key == 't':
                continue

            if key[len(key)-1] == "*": #Get key value, without the * if it is there
                continue
            else:
                tok = key

            if numArbors[tok] > 1:
                continue
            arborID = 0

            if key == 'a':
                countActivity(data,key)

            for preNeuronID in range(numPreNeurons[tok]):
                plotMe = array(data[key][arborID][preNeuronID][:])
                if len(plotMe) != 0:
                    #Special cases for legend labels on printing
                    if "_" in key: #Specific pre-neuron and conn is given
                        keySP       = key.split("_")
                        keyLabel    = keySP[0]
                        arborLabel  = keySP[1]
                        neuronLabel = keySP[2]
                        figLabel=keyLabel+"_"+filenameTup[0]+"_n"+neuronLabel+"_a"+arborLabel
                    else:
                        keyLabel = key
                        arborLabel = str(arborID)
                        if key[len(key)-1] == "*": #preNeuron
                            neuronLabel = 'Avg'
                            figLabel=keyLabel+"_"+filenameTup[0]+"_n"+neuronLabel+"_a"+arborLabel
                        else:
                            neuronLabel = 'Post'
                            figLabel=keyLabel+"_"+filenameTup[0]+"_n"+neuronLabel

                    if '_1_' in key:
                        plot(time, plotMe, ':',label=figLabel)
                    else:
                        plot(time, plotMe,label=figLabel)
                    grid(True)
                    didPlot = True

        if didPlot:
            if doLegend:
                legend()#bbox_to_anchor=(0., 1.02, 1., .102), ncol = 2, mode="expand", borderaxespad=0.,loc=3)
            tight_layout()
            if not dispFigs:
                savefig(figOutDir+rootFigName+"_"+filenameTup[0]+"_timeCourse.png")
                clf()

    #Clear lines for this file
    #del lines #Will not free until garabe collection

if dispFigs:
    show()

print "\nreadProbe: Script Complete...\n"
