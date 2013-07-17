import numpy as np
import pylab as pl
from matplotlib.mlab import normpdf
from writePvpFile import writeHeaderFile, writeData
from math import pi, sqrt
import os

#For plotting
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

datasetVal = 1
#eyeVal = 1
numBins = 32
numSigma = 2

targetMean = 0
targetStd = 1

depthFileDir = "/nh/compneuro/Data/Depth/depth_data_"+str(datasetVal)+"/"
depthFileListDir = depthFileDir + "list/"
outputFileDir = "/nh/compneuro/Data/Depth/depth_data_"+str(datasetVal)+"/pvp/"
if not os.path.exists(depthFileListDir):
   os.makedirs(depthFileListDir)

if not os.path.exists(outputFileDir):
   os.makedirs(outputFileDir)

for eyeVal in range(2):
   depthFileList = depthFileListDir + "depth_0" + str(eyeVal) + ".txt"
   outputFileName = outputFileDir + "depth_0" + str(eyeVal) + ".pvp"
#Open output file
   outMatFile = open(outputFileName, 'wb')

   tempFile = open(depthFileList, 'r')
   fileList = tempFile.readlines()
   tempFile.close()

#Remove newlines
   fileList = [file[:len(file)-1] for file in fileList]

   numFrames = len(fileList)
#Grab image size
   image = pl.imread(fileList[0])
   (Y, X) = np.shape(image)

#Write out header
   writeHeaderFile(outMatFile, (Y, X, numBins), numFrames)

#Calculate stepSize based on number of bins
   stepSize = float(1)/numBins
   normVal = 1/(stepSize * sqrt(2 * pi))

#depthFile = fileList[0]
   for frameIdx, depthFile in enumerate(fileList):
      print "file", depthFile
      image = pl.imread(depthFile)
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

