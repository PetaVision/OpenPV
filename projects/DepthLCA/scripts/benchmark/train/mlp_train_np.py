import os, sys
lib_path = os.path.abspath("/home/slundquist/workspace/PetaVision/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon
from plotReconError import plotReconError

#For plotting
#import matplotlib.pyplot as plt

outputDir = "/nh/compneuro/Data/Depth/LCA/benchmark/stereo_train_mlp/"
skipFrames = 1
doPlotRecon = True
doPlotErr = False
errShowPlots = False
layers = [
   "a3_DepthImage",
   "a10_Est",
   ]
#Layers for constructing recon error
preErrLayers = [
   "a2_LeftRescale",
   "a2_LeftRescale",
   "a2_LeftRescale",
   "a7_LeftReconAll",
   "a7_LeftReconAll",
   "a7_LeftReconAll",
]

postErrLayers = [
   "a4_LeftReconS2",
   "a5_LeftReconS4",
   "a6_LeftReconS8",
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
   .0294,
   .0294,
   .0294,
   1,
   1,
   1,
]


if(doPlotRecon):
   print "Plotting reconstructions"
   plotRecon(layers, outputDir, skipFrames, 0, False)

if(doPlotErr):
   print "Plotting reconstruction error"
   plotReconError(preErrLayers, postErrLayers, preToPostScale, outputDir, errShowPlots, skipFrames, gtLayers)
