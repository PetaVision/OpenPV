import os, sys
lib_path = os.path.abspath("/home/slundquist/workspace/PetaVision/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon
from plotReconError import plotReconError
#import numpy as np
#from readPvpFile import readHeaderFile, readData, toFrame
#from scipy.misc import imsave
#from pylab import *
##Debugging
#import pdb

outputDir = "/nh/compneuro/Data/Depth/LCA/depth_log_scale_ndepth/"
skipFrames = 200#Only print every 20th frame
doPlotRecon = True
doPlotErr = False
errShowPlots = True
layers = [
   "a10_LeftRescale1",
   "a11_LeftRescale2",
   "a12_LeftRescale3",
   "a16_LeftRecon1",
   "a17_LeftRecon2",
   "a18_LeftRecon3",
   "a29_RightRescale1",
   "a30_RightRescale2",
   "a31_RightRescale3",
   "a35_RightRecon1",
   "a36_RightRecon2",
   "a37_RightRecon3",
]

#Layers for constructing recon error
preErrLayers = [
   "a10_LeftRescale1",
   "a11_LeftRescale2",
   "a12_LeftRescale3",
   "a29_RightRescale1",
   "a30_RightRescale2",
   "a31_RightRescale3",
]
postErrLayers = [
   "a16_LeftRecon1",
   "a17_LeftRecon2",
   "a18_LeftRecon3",
   "a35_RightRecon1",
   "a36_RightRecon2",
   "a37_RightRecon3",
]

preToPostScale = [
   .0624,
   .0624,
   .0588,
   .0624,
   .0624,
   .0588,
]


if(doPlotRecon):
   plotRecon(layers, outputDir, skipFrames)

if(doPlotErr):
   plotReconError(preErrLayers, postErrLayers, preToPostScale, outputDir, errShowPlots)
