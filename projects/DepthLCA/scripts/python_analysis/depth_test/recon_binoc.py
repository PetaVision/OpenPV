import os, sys
lib_path = os.path.abspath("/home/slundquist/workspace/PetaVision/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon
from plotReconError import plotReconError

#For plotting
#import matplotlib.pyplot as plt

outputDir = "/nh/compneuro/Data/Depth/LCA/test_stack_nowhite/"
skipFrames = 1 #Only print every 20th frame
doPlotRecon = True
doPlotErr = False
errShowPlots = False
layers = [
   "a2_LeftRescale",
   "a4_LeftReconS2",
   "a5_LeftReconS4",
   "a6_LeftReconS8",
   "a7_LeftReconAll",
   "a10_RightRescale",
   "a11_RightReconS2",
   "a12_RightReconS4",
   "a13_RightReconS8",
   "a14_RightReconAll",
]
#Layers for constructing recon error

preErrLayers = [
   "a3_LeftRescale",
   "a3_LeftRescale",
   "a3_LeftRescale",
   "a3_LeftRescale",
   "a12_RightRescale",
   "a12_RightRescale",
   "a12_RightRescale",
   "a12_RightRescale",
]
postErrLayers = [
   "a5_LeftReconS2",
   "a6_LeftReconS4",
   "a7_LeftReconS8",
   "a8_LeftReconAll",
   "a13_RightReconS2",
   "a14_RightReconS4",
   "a15_RightReconS8",
   "a16_RightReconAll",
]

normalizeIdx = [-1, 0, 0, 0, -1, 4, 4, 4]

preToPostScale = [
   .0294,
   .0294,
   .0294,
   .0294,
   .0294,
   .0294,
   .0294,
   .0294,
]

if(doPlotRecon):
   print "Plotting reconstructions"
   plotRecon(layers, outputDir, skipFrames)

if(doPlotErr):
   print "Plotting reconstruction error"
   plotReconError(preErrLayers, postErrLayers, preToPostScale, outputDir, errShowPlots, normalizeIdx, skipFrames)
