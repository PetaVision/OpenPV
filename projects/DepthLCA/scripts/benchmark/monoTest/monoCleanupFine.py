import os, sys
lib_path = os.path.abspath("/home/ec2-user/workspace/pv-core/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon
from plotReconError import plotReconError

#For plotting
#import matplotlib.pyplot as plt

outputDir = "/home/ec2-user/mountData/benchmark/icaweights_bugged/monoTest/cleanup_fine/"
skipFrames = 1 #Only print every 20th frame
doPlotRecon = True
doPlotErr = False
errShowPlots = False

layers = [
   "a1_LeftImage",
   "a2_RightImage",
   "a3_LeftRecon",
   "a5_RightRecon",
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
   print("Plotting reconstructions")
   plotRecon(layers, outputDir, skipFrames)

if(doPlotErr):
   print("Plotting reconstruction error")
   plotReconError(preErrLayers, postErrLayers, preToPostScale, outputDir, errShowPlots, skipFrames, gtLayers)
