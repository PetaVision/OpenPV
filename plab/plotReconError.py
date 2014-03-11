import os, sys
import numpy as np
from readPvpFile import readHeaderFile, readData, toFrame
from pylab import *

def plotReconError(preErrLayers, postErrLayers, preToPostScale, outputDir, showPlots, skipFrames=0):
   if(len(preErrLayers) != len(postErrLayers)):
      print "Pre and post error layers not the same length"
      sys.exit();
   if(len(preErrLayers) != len(preToPostScale)):
      print "PreToPostScale must be same length as pre and post err layers"
      sys.exit();

   errDir = outputDir + "ErrVsTime/"
   if not os.path.exists(errDir):
      os.makedirs(errDir)
   for (preErr, postErr, scale) in zip(preErrLayers, postErrLayers, preToPostScale):
      prePvpFile = open(outputDir + preErr + ".pvp", 'rb')
      postPvpFile = open(outputDir + postErr + ".pvp", 'rb')
      #Grab header
      preHeader = readHeaderFile(prePvpFile)
      postHeader = readHeaderFile(postPvpFile)

      if(preHeader["ny"] != postHeader["ny"] or preHeader["nx"] != postHeader["nx"] or preHeader["nf"] != preHeader["nf"]):
         print "pre layer " + preErr + " and post layer " + postErr + " size not the same"
         sys.exit()

      shape = (preHeader["ny"], preHeader["nx"], preHeader["nf"])
      numPerFrame = shape[0] * shape[1] * shape[2]
      #Read until errors out (EOF)
      (preIdx, preMat) = readData(prePvpFile, shape, numPerFrame)
      (postIdx, postMat) = readData(postPvpFile, shape, numPerFrame)
      idx = []
      datapts = []
      while preIdx != -1 and postIdx != -1:
         idx.append(preIdx)
         #Find average of error layer and add data point
         diff = np.std((preMat*scale) - postMat) / np.std(preMat*scale)
         datapts.append(np.mean(diff))
         #Read a few extra for skipping frames
         for i in range(skipFrames):
            (preIdx, preMat) = readData(prePvpFile, shape, numPerFrame)
            (postIdx, postMat) = readData(postPvpFile, shape, numPerFrame)
            if(preIdx == -1 or postIdx == -1):
                break

      #plot datapts
      figure()
      plot(idx, datapts)
      title(preErr + " - " + postErr + " error over time");
      savefig(errDir + preErr + "_" + postErr + "_err_vs_time.png");
      if showPlots:
         show()
