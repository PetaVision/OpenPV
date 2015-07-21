import os, sys
lib_path = os.path.abspath("/home/slundquist/workspace/PetaVision/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon
from plotReconError import plotReconError

#For plotting
#import matplotlib.pyplot as plt

outputDir = "/nh/compneuro/Data/Depth/LCA/benchmark/stereo_one_eye_train/"
startFrame = 10
skipFrames = 11 #Only print every 20th frame
doPlotRecon = True
doPlotErr = True
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
   #"a23_ForwardLayer",
   #"a26_DepthGT",
   #"a28_RCorrReconS2",
   #"a29_RCorrReconS4",
   #"a30_RCorrReconS8",
   #"a31_RCorrReconAll",
   ]
#Layers for constructing recon error
preErrLayers = [
   "a2_LeftRescale",
   #"a26_DepthGT",
   #"a26_DepthGT",
   #"a31_RCorrReconAll",
   #"a31_RCorrReconAll",
   #"a31_RCorrReconAll",
]

postErrLayers = [
   "a7_LeftReconAll",
   #"a23_ForwardLayer",
   #"a31_RCorrReconAll",
   #"a28_RCorrReconS2",
   #"a29_RCorrReconS4",
   #"a30_RCorrReconS8",
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
   #.0294,
   #.0294,
   #.0294,
   #1,
   #1,
   #1,
   #1,
   #1,
   #1,
   #1,
   #1,
]


if(doPlotRecon):
   print "Plotting reconstructions"
   plotRecon(layers, outputDir, skipFrames, startFrame)

if(doPlotErr):
   print "Plotting reconstruction error"
   plotReconError(preErrLayers, postErrLayers, preToPostScale, outputDir, errShowPlots, skipFrames, gtLayers)
