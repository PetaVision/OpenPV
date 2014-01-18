import numpy as np
import pylab as pl
from matplotlib.mlab import normpdf
from writePvpFile import writeHeaderFile, writeData
from math import pi, sqrt
import os



#For plotting
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

#datasetValList = [1, 2, 5, 9, 11, 13, 14, 17, 18, 48, 51, 56, 57, 59, 60, 84, 91, 93, 95, 96, 104, 106, 113, 117]
datasetValList = [1]

#eyeVal = 1
numBins = 32
numSigma = 2

targetMean = 0
targetStd = 1

#outputFileDir = "/nh/compneuro/Data/Depth/concat_data_temp/"
outputFileDir = "/tmp/"

#if not os.path.exists(depthFileListDir):
#   os.makedirs(depthFileListDir)
#
if not os.path.exists(outputFileDir):
   os.makedirs(outputFileDir)


for eyeVal in range(2):
    outputImageList = outputFileDir + "image_0" + str(eyeVal)
    #If it exist, remove
    try:
        os.remove(outputImageList)
    except OSError:
        pass

    #Create a file list of depth files to make
    fileList = []
    print "Compiling File List"
    for datasetVal in datasetValList:
        #Set filenames
        depthFileDir = "/nh/compneuro/Data/Depth/depth_data_"+str(datasetVal)+"/"
        depthFileListDir = depthFileDir + "list/"

        #Create list files if they don't exist
        if not os.path.exists(depthFileListDir):
            os.makedirs(depthFileListDir)

        depthFileList = depthFileListDir + "depth_0" + str(eyeVal) + ".txt"
        try:
            with open(depthFileList):
                pass
        except IOError:
            depthDir = depthFileDir + "depth_0" + str(eyeVal) + "/"
            #Make the depth list file
            os.system("ls " + depthDir + "* -1d > " + depthFileList);

        imageFileList = depthFileListDir + "image_0" + str(eyeVal) + ".txt"
        imageDir = depthFileDir + "image_0" + str(eyeVal) + "/"
        try:
            with open(imageFileList):
                pass
        except IOError:
            #Make the image list file
            os.system("ls " + imageDir + "* -1d > " + imageFileList);
        #Append imageFileList to the concat final image list
        os.system("ls " + imageDir + "* -1d >> " + outputImageList)

        print depthFileList

        tempFile = open(depthFileList, 'r')
        #Add to list
        fileList.extend(tempFile.readlines())
        tempFile.close()
    #Remove newlines
    fileList = [file[:len(file)-1] for file in fileList]
    #Grab image size
    image = pl.imread(fileList[0])
    (Y, X) = np.shape(image)
    #Grab total number of frames
    numFrames = len(fileList)
    #Set output filename
    outputFileName = outputFileDir + "depth_0" + str(eyeVal) + ".pvp"
    #Open output file
    outMatFile = open(outputFileName, 'wb')
    #Write out header
    writeHeaderFile(outMatFile, (Y, X, numBins), numFrames)
    #Calculate stepSize based on number of bins
    stepSize = float(1)/numBins
    normVal = 1/(stepSize * sqrt(2 * pi))
    for frameIdx, depthFile in enumerate(fileList):
        print "Creating file", depthFile
        image = pl.imread(depthFile)
        #Make sure the size is the same
        (Ytest, Xtest) = np.shape(image)
        assert(Y == Ytest and X == Xtest)
        depthMat = np.zeros((Y, X, numBins))
        binImage = np.zeros((Y, X, numBins))
        for bin in range(numBins):
            binupthresh = stepSize * (bin + 1)
            binlowthresh = stepSize * (bin)
            mat1 = image < binupthresh
            mat2 = image >= binlowthresh
            by, bx = (mat1 * mat2).nonzero()
            binImage[by,bx,bin] = 1
            for binRange in range(-numSigma, numSigma+1):
                curBin = bin + binRange
                #Check boundary conditions
                if curBin < 0 or curBin >= numBins:
                    continue
                upthresh = stepSize * (curBin + 1)
                lowthresh = stepSize * (curBin)
                #Calculate cumulative distribution
                #Using center of stepsize
                xVal = lowthresh + (float(upthresh - lowthresh)/2)
                outmat = (normpdf(xVal, image, stepSize)) * binImage[:,:,bin]
                depthMat[:, :, curBin] += outmat[:,:]
        #Normalize with max as 1
        #depthMat = depthMat / normVal
        #Normalize with mean/std
        matMean = np.mean(depthMat)
        matStd = np.std(depthMat)
        depthMat = (depthMat - matMean) * (targetStd/matStd) + targetMean
        #Write data for frame
        writeData(outMatFile, depthMat, frameIdx)
    outMatFile.close()

#y, x, z = depthMat.nonzero()
#
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(x, z, -y, zdir = 'z', color = 'r')
#plt.show()

