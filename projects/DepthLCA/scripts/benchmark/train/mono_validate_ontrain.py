import os, sys
lib_path = os.path.abspath("/home/slundquist/workspace/PetaVision/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon
from plotReconError import plotReconError

#For plotting
#import matplotlib.pyplot as plt

outputDir = "/nh/compneuro/Data/Depth/LCA/benchmark/mono_validate_rcorr_batch_ontrain/"
skipFrames = 1 #Only print every 20th frame
doPlotRecon = True
doPlotErr = False
errShowPlots = False
layers = [
   "a3_LeftReconS2",
   "a4_LeftReconS4",
   "a5_LeftReconS8",
   "a6_LeftReconAll",
   "a7_DepthImage",
   "a12_RCorrReconS2",
   "a13_RCorrReconS4",
   "a14_RCorrReconS8",
   "a15_RCorrReconAll",
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
