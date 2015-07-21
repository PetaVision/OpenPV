import os, sys
lib_path = os.path.abspath("/home/slundquist/workspace/PetaVision/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon
from plotReconError import plotReconError

#For plotting
#import matplotlib.pyplot as plt

outputDir = "/nh/compneuro/Data/Depth/LCA/Checkpoints/saved_stack_v2/"
skipFrames = 1 #Only print every 20th frame
doPlotRecon = True
doPlotErr = False
errShowPlots = False
layers = [
   "DepthRescale_A",
   "DepthReconS2_A",
   "DepthReconS4_A",
   "DepthReconS8_A",
   "DepthReconAll_A"
   ]
#Layers for constructing recon error
preErrLayers = [
   #"a3_LeftRescale",
   #"a3_LeftRescale",
   #"a3_LeftRescale",
   "a8_LeftReconAll",
   "a8_LeftReconAll",
   "a8_LeftReconAll",
]

postErrLayers = [
   #"a5_LeftReconS2",
   #"a6_LeftReconS4",
   #"a7_LeftReconS8",
   "a5_LeftReconS2",
   "a6_LeftReconS4",
   "a7_LeftReconS8",
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



#Not used, todo

preToPostScale = [
   #.0294,
   #.0294,
   #.0294,
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
