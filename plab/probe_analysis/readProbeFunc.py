from numpy import *

def readProbeFile(filename,startTime,endTime):
    f = open(filename, 'r')
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
    return lines


def splitLine(line):
   #Split line by :
   lineSp = line.split(":")
   assert(len(lineSp) == 2) ##only one colon in params name TODO: this error is thrown if start & end times are way above probe times - why?
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


def getNumArbors(tok,line):
    #items[0] is the tok, items[1] is the arbor number, items[2] is the patchNeuronNum
    tempList = [items[1] if items[0] == tok and len(items) > 1 else '0' for items in [part[0].split('_') for part in line]]
    return max([int(num) for num in tempList])+1
        

def countActivity(data,key):
   print "Average activity: " + str(len(nonzero(data[key])[0]))
