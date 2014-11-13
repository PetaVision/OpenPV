import os, sys
lib_path = os.path.abspath("/home/slundquist/workspace/PetaVision/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon
from plotReconError import plotReconError

#For plotting
#import matplotlib.pyplot as plt

outputDir = "/nh/compneuro/Data/MLPLCA/LCA/depth_slp/"
skipFrames = 1
doPlotRecon = True
doPlotErr = False
errShowPlots = False
layers = [
   "a5_DepthGT",
   "a6_ForwardLayer"
   ]

if(doPlotRecon):
   print "Plotting reconstructions"
   plotRecon(layers, outputDir, skipFrames)

#if(doPlotErr):
#   print "Plotting reconstruction error"
#   plotReconError(preErrLayers, postErrLayers, preToPostScale, outputDir, errShowPlots, skipFrames, gtLayers)
