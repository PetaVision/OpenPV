import numpy as np
from readPvpFile import readHeaderFile, readData
from scipy.misc import imsave
import os

#For plotting
#import matplotlib.pyplot as plt

#datasetVal = 2
#eyeVal = 1
#depthFileListDir = "/nh/compneuro/Data/Depth/depth_data_"+str(datasetVal) + "/list/"
#depthFileList = depthFileListDir + "depth_0" + str(eyeVal) + ".txt"
#pvpFileName = "/nh/compneuro/Data/Depth/depth_data_1/pvp/depth_0" + str(eyeVal)+ ".pvp"
#lastCheckpoint = 680000
outputDir = "/nh/compneuro/Data/Depth/LCA/binoc_pos_2sf/"
#outputDir = "/nh/compneuro/Data/Depth/LCA/dataset02/"
readFromCheckpoint = False
layers = [
      "a2_LeftGanglion1",
      "a3_LeftGanglion2",
      "a4_LeftDownsample1",
      "a5_LeftDownsample2",
      "a6_LeftRescale1",
      "a7_LeftRescale2",
      "a8_LeftError1",
      "a9_LeftError2",
      "a10_LeftRecon1",
      "a11_LeftRecon2",
      "a14_RightGanglion1",
      "a15_RightGanglion2",
      "a16_RightDownsample1",
      "a17_RightDownsample2",
      "a18_RightRescale1",
      "a19_RightRescale2",
      "a20_RightError1",
      "a21_RightError2",
      "a22_RightRecon1",
      "a23_RightRecon2",
      "a25_PosDownsample1",
      "a26_PosDownsample2",
      "a27_PosError1",
      "a28_PosError2",
      "a29_PosRecon1",
      "a30_PosRecon2",
      ]

#checkpointDir = outputDir + "Checkpoints/Checkpoint"+str(lastCheckpoint)+"/"

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
   (Y, X, Z) = np.shape(mat)
   assert Z == 1
   img = mat[:, :, 0]
   img = (img - np.min(img)) / (np.max(img) - np.min(img))
   return img


reconDir = outputDir + "Recon/"
if not os.path.exists(reconDir):
   os.makedirs(reconDir)


#Open file
for layername in layers:
#layername = layers[0]
   if readFromCheckpoint:
      pvpFile = open(checkpointDir + layername + ".pvp", 'rb')
   else:
      pvpFile = open(outputDir + layername + ".pvp", 'rb')

   #Grab header
   header = readHeaderFile(pvpFile)
   shape = (header["ny"], header["nx"], header["nf"])
   numPerFrame = shape[0] * shape[1] * shape[2]

   if readFromCheckpoint:
      #Read only one timestamp
      (idx, mat) = readData(pvpFile, shape, numPerFrame)
      if header["nf"] > 1:
         img = matToImage(mat)
      else:
         img = scaleMat(mat)
      imsave(reconDir + layername + ".png", img)
   else:
      #Read until errors out (EOF)
      (idx, mat) = readData(pvpFile, shape, numPerFrame)
      #While not eof
      while idx != -1:
         if header["nf"] > 1:
            img = matToImage(mat)
         else:
            img = scaleMat(mat)
         imsave(reconDir + layername + str(int(idx[0])) + ".png", img)
         (idx, mat) = readData(pvpFile, shape, numPerFrame)


#plt.imshow(img)
#plt.show()
