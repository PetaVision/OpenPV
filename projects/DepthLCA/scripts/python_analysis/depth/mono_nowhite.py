import os, sys
lib_path = os.path.abspath("/home/slundquist/workspace/PetaVision/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon
from plotReconError import plotReconError

#For plotting
#import matplotlib.pyplot as plt

outputDir = "/nh/compneuro/Data/Depth/LCA/stack_mono_nowhite/"
skipFrames = 100 #Only print every 20th frame
doPlotRecon = True
doPlotErr = True
errShowPlots = False
layers = [
   "a2_LeftRescale",
   "a4_LeftReconS2",
   "a5_LeftReconS4",
   "a6_LeftReconS8",
   "a7_LeftReconAll",
   ]
#Layers for constructing recon error
preErrLayers = [
   "a7_LeftReconAll",
   "a7_LeftReconAll",
   "a7_LeftReconAll",
]

postErrLayers = [
   "a4_LeftReconS2",
   "a5_LeftReconS4",
   "a6_LeftReconS8",
]

gtLayers = None
#gtLayers = [
#   #"a25_DepthRescale",
#   #"a25_DepthRescale",
#   #"a25_DepthRescale",
#   "a25_DepthRescale",
#   "a25_DepthRescale",
#   "a25_DepthRescale",
#]

preToPostScale = [
   1,
   1,
   1,
]


if(doPlotRecon):
   print "Plotting reconstructions"
   plotRecon(layers, outputDir, skipFrames)

if(doPlotErr):
   print "Plotting reconstruction error"
   plotReconError(preErrLayers, postErrLayers, preToPostScale, outputDir, errShowPlots, skipFrames, gtLayers)
