import os, sys
import numpy as np
from readPvpFile import readHeaderFile, readData, toFrame
#from scipy.misc import imsave
from pylab import *
import matplotlib.pyplot as plt
import pdb

def plotReconError(preErrLayers, postErrLayers, preToPostScale, outputDir, showPlots, skipFrames=1, gtLayers=None, gtThresh=.7) :
   if(len(preErrLayers) != len(postErrLayers)):
      print("Pre and post error layers not the same length")
      sys.exit();
   if(len(preErrLayers) != len(preToPostScale)):
      print("PreToPostScale must be same length as pre and post err layers")
      sys.exit();

   if(gtLayers == None):
      plotErrVsGt = False
   else:
      plotErrVsGt = True
      if len(gtLayers) != len(preErrLayers):
         print("gtLayers must be same length as pre and post err layers")
         sys.exit();

   errDir = outputDir + "ErrVsTime/"
   if not os.path.exists(errDir):
      os.makedirs(errDir)

   errHeatDir = outputDir + "ErrHeatmap/"
   if not os.path.exists(errHeatDir):
      os.makedirs(errHeatDir)

   if plotErrVsGt:
      errVsGtDir = outputDir + "ErrVsGt/"
      if not os.path.exists(errVsGtDir):
         os.makedirs(errVsGtDir)

   idxs = {}
   data = {}
   for (i, (preErr, postErr, scale)) in enumerate(zip(preErrLayers, postErrLayers, preToPostScale)):
      prePvpFile = open(outputDir + preErr + ".pvp", 'rb')
      postPvpFile = open(outputDir + postErr + ".pvp", 'rb')
      #Grab header info
      preHeader = readHeaderFile(prePvpFile)
      postHeader = readHeaderFile(postPvpFile)
      if(preHeader["ny"] != postHeader["ny"] or preHeader["nx"] != postHeader["nx"] or preHeader["nf"] != preHeader["nf"]):
         print("pre layer " + preErr + " and post layer " + postErr + " size not the same")
         sys.exit()
      shape = (preHeader["ny"], preHeader["nx"], preHeader["nf"])
      numPerFrame = shape[0] * shape[1] * shape[2]

      if(plotErrVsGt):
         gtLayerFile = open(outputDir + gtLayers[i] + ".pvp", 'rb')
         gtHeader = readHeaderFile(gtLayerFile)
         gtShape = (gtHeader["ny"], gtHeader["nx"], gtHeader["nf"])
         gtNumPerFrame = gtShape[0] * gtShape[1] * gtShape[2]
         #Calculate difference in shape
         gtXScale = float(gtHeader["nx"]) / preHeader["nx"]
         gtYScale = float(gtHeader["ny"]) / preHeader["ny"]

      #Read until errors out (EOF)
      (preIdx, preMat) = readData(prePvpFile, shape, numPerFrame)
      (postIdx, postMat) = readData(postPvpFile, shape, numPerFrame)
      if(plotErrVsGt):
         (gtIdx, gtMat) = readData(gtLayerFile, gtShape, gtNumPerFrame)
         gtpts = []

      iidx = []
      idatapts = []
      while preIdx != -1 and postIdx != -1:
         print(preErr + " to " + postErr + ": " + str(int(preIdx[0])))
         iidx.append(preIdx)
         #Find average of error layer and add data point
         diff = np.std((preMat*scale) - postMat) / np.std(preMat*scale)
         idatapts.append(np.mean(diff))
         #plt.pcolor(heatImg)
         #plt.axis([0, x, 0, y])
         #plt.colorbar()
         #savefig(errHeatDir + preErr + "_" + postErr + "_err_heatmap.png");

         #plt.show()

         #heatImg = postMat / (preMat*scale)
         #postMat is current layer, preMat is all
         heatImg = 1 - ((preMat*scale - postMat)/(preMat*scale));
         heatImg = np.squeeze(heatImg)
         #Scale from 0 to 1
         #heatImg = (heatImg - np.min(heatImg))/(np.max(heatImg) - np.min(heatImg))
         #[y, x] = np.shape(heatImg)

         plt.imsave(errHeatDir + preErr + "_" + postErr + str(int(preIdx[0])) + ".png", heatImg, vmin=0, vmax=1)

         if(plotErrVsGt):
            #Need to make data points per pixel, with different scales for the gt
            #Only taking care of the case where gt is smaller than heatmap
            assert(gtXScale <= 1)
            assert(gtYScale <= 1)
            #Only doing depth maps right now
            assert(gtHeader["nf"] > 1)
            #Grab image mat from 3d matrix
            flatGtMat = np.argmax(gtMat, 2)
            #Expand matrix to size of heatImg
            expandScale = np.ones((1/gtYScale, 1/gtXScale))
            expGtMat = np.kron(flatGtMat, expandScale)
            onIdxs = np.nonzero(heatImg >= gtThresh)

            assert(np.shape(expGtMat) == np.shape(heatImg))
            gtpts = np.concatenate((gtpts, expGtMat[onIdxs].flatten(1)))

            #pdb.set_trace()
            #Find all instances above threshold

            #errpts = np.concatenate((errpts, heatImg.flatten(1)))
            #gtpts = np.concatenate((gtpts, expGtMat.flatten(1)))

         #Read a few extra for skipping frames
         for skipi in range(skipFrames):
            (preIdx, preMat) = readData(prePvpFile, shape, numPerFrame)
            (postIdx, postMat) = readData(postPvpFile, shape, numPerFrame)
            if(preIdx == -1 or postIdx == -1):
                break
            if(plotErrVsGt):
               (gtIdx, gtMat) = readData(gtLayerFile, gtShape, gtNumPerFrame)
               if(gtIdx == -1):
                  break
      #End for frames
      prePvpFile.close()
      postPvpFile.close()
      idxs[i] = iidx
      data[i] = idatapts
      if(plotErrVsGt):
         gtLayerFile.close()

      #plot datapts
      figure()
      plot(iidx, idatapts)
      title(preErr + " - " + postErr + " error over time");
      savefig(errDir + preErr + "_" + postErr + "_err_vs_time.png");
      if showPlots:
         show()

      if plotErrVsGt:
         figure()
         hist(gtpts, 5)
         title(preErr + " - " + postErr + " vs gt");
         savefig(errVsGtDir + preErr + "_" + postErr + "_vs_gt.png");
         if showPlots:
            show()
   #End for each layer combo
#End function

