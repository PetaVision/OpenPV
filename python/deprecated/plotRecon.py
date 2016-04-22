import os, sys
import numpy as np
from readPvpFile import readHeaderFile, readData, toFrame
from scipy.misc.pilutil import imsave
import pdb

def matToImage(mat):
   (Y, X, Z) = np.shape(mat)
   #Get stepsize
   stepSize = float(1)/Z
   #Grab max value of bins
   maxMat = np.max(mat, 2)
   #Tile maxmat into shape of origonal mat
   maxMat = np.tile(maxMat, (Z, 1, 1))
   #Change back into origonal shape
   maxMat = np.swapaxes(maxMat, 0, 1)
   maxMat = np.swapaxes(maxMat, 1, 2)
   #Grab indicides of matrix where it matches
   m1 = mat != 0
   m2 = mat == maxMat
   bolMat = m1*m2
   (yidx, xidx, zidx) = bolMat.nonzero()
   outimg = np.zeros((Y, X))
   upthresh = stepSize*(zidx+1)
   lowthresh = stepSize*zidx
   idxVal = lowthresh + (upthresh - lowthresh)/2
   outimg[yidx, xidx] = idxVal
   return outimg

#Scales mat to be between 0 and 1 for image saving
def scaleMat(mat):
   img = mat
   img = (img - np.min(img)) / (np.max(img) - np.min(img))
   return img.squeeze()


##
# A function to plot reconstructions of given layers
# @layernames A list of layer names (without the .pvp after) to reconstruct
# @outputDir The directory where the pvp files are located, as well as the directory to store output plots
# @skipFrames Number of frames to skip in reconstructions
##
def plotRecon(layernames, dataDir, skipFrames, startFrames=0, scale=True, outputDir=None, suffix=""):
   if outputDir == None:
      outputDir = dataDir
   reconDir = outputDir + "Recon/"
   if not os.path.exists(reconDir):
      os.makedirs(reconDir)

   #Open file
   for layername in layernames:
      pvpFile = open(dataDir + layername + ".pvp", 'rb')

      #Grab header
      header = readHeaderFile(pvpFile)
      shape = (header["ny"], header["nx"], header["nf"])
      numPerFrame = shape[0] * shape[1] * shape[2]

      #Read until start frames
      pvpFile.seek((numPerFrame*4+8)*startFrames,1)

      (idx, mat) = readData(pvpFile, shape, numPerFrame)
      #if idx != -1:
      #   #Read until errors out (EOF)
      #   for i in range(startFrames):
      #      (idx, mat) = readData(pvpFile, shape, numPerFrame)
      #      if(idx == -1):
      #         break

      #While not eof
      while idx != -1:
         print(layername + ": " + str(int(idx[0])))
         #color bands
         if header["nf"] == 1 or header["nf"] == 3:
            if(scale):
               img = scaleMat(mat)
            else:
               img = (np.uint8)(mat.squeeze()*256)
         else:
            img = matToImage(mat)
         imsave(reconDir + layername + str(int(idx[0])) + suffix + ".png", img)
         #Read a few extra for skipping frames
         for i in range(skipFrames):
             (idx, mat) = readData(pvpFile, shape, numPerFrame)
             if(idx == -1):
                 break
      pvpFile.close()
