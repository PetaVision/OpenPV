import os, sys
lib_path = os.path.abspath("/home/slundquist/workspace/PetaVision/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon
from plotReconError import plotReconError

#For plotting
#import matplotlib.pyplot as plt

outputDir = "/nh/compneuro/Data/Depth/LCA/stack_with_mono_gpu/"
skipFrames = 20 #Only print every 20th frame
doPlotRecon = True
doPlotErr = True
errShowPlots = False
layers = [
   "a2_LeftRescale",
   "a4_LeftReconS2",
   "a5_LeftReconS4",
   "a6_LeftReconS8",
   "a7_monoLeftReconS2",
   "a8_monoLeftReconS4",
   "a9_monoLeftReconS8",
   "a10_LeftReconAll",
   "a13_RightRescale",
   "a15_RightReconS2",
   "a16_RightReconS4",
   "a17_RightReconS8",
   "a18_monoRightReconS2",
   "a19_monoRightReconS4",
   "a20_monoRightReconS8",
   "a21_RightReconAll",
   ]
#Layers for constructing recon error
preErrLayers = [
   "a10_LeftReconAll",
   "a10_LeftReconAll",
   "a10_LeftReconAll",
   "a10_LeftReconAll",
   "a10_LeftReconAll",
   "a10_LeftReconAll",
]

postErrLayers = [
   "a4_LeftReconS2",
   "a5_LeftReconS4",
   "a6_LeftReconS8",
   "a7_monoLeftReconS2",
   "a8_monoLeftReconS4",
   "a9_monoLeftReconS8",
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
   plotRecon(layers, outputDir, skipFrames)

if(doPlotErr):
   print "Plotting reconstruction error"
   plotReconError(preErrLayers, postErrLayers, preToPostScale, outputDir, errShowPlots, skipFrames, gtLayers)
