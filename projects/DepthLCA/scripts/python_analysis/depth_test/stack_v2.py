import os, sys
lib_path = os.path.abspath("/home/slundquist/workspace/PetaVision/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon
from plotReconError import plotReconError

#For plotting
#import matplotlib.pyplot as plt

outputDir = "/nh/compneuro/Data/Depth/LCA/test_stack_v2/"
skipFrames = 1 #Only print every 20th frame
doPlotRecon = False
doPlotErr = True
errShowPlots = False
layers = [
   "a3_LeftRescale",
   "a5_LeftReconS2",
   "a6_LeftReconS4",
   "a7_LeftReconS8",
   "a8_LeftReconAll",
   "a12_RightRescale",
   "a14_RightReconS2",
   "a15_RightReconS4",
   "a16_RightReconS8",
   "a17_RightReconAll",
   "a25_DepthRescale",
   "a26_DepthReconS2",
   "a27_DepthReconS4",
   "a28_DepthReconS8",
   "a29_DepthReconAll",
   #"a37_V2V1S2LeftRecon",
   #"a38_V2V1S4LeftRecon",
   #"a39_V2V1S8LeftRecon",
   #"a40_V2V1AllLeftRecon",
   #"a41_V2V1S2RightRecon",
   #"a42_V2V1S4RightRecon",
   #"a43_V2V1S8RightRecon",
   #"a44_V2V1AllRightRecon"
   ]
#Layers for constructing recon error
preErrLayers = [
   "a3_LeftRescale",
   "a3_LeftRescale",
   "a3_LeftRescale",
   "a3_LeftRescale",
   "a3_LeftRescale",
   "a3_LeftRescale",
   "a3_LeftRescale",
   "a3_LeftRescale",

   "a12_RightRescale",
   "a12_RightRescale",
   "a12_RightRescale",
   "a12_RightRescale",
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
   "a37_V2V1S2LeftRecon",
   "a38_V2V1S4LeftRecon",
   "a39_V2V1S8LeftRecon",
   "a40_V2V1AllLeftRecon",


   "a14_RightReconS2",
   "a15_RightReconS4",
   "a16_RightReconS8",
   "a17_RightReconAll",
   "a41_V2V1S2RightRecon",
   "a42_V2V1S4RightRecon",
   "a43_V2V1S8RightRecon",
   "a44_V2V1AllRightRecon"
]

#Not used, todo
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
