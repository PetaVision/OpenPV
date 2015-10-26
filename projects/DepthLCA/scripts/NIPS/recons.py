import os, sys
lib_path = os.path.abspath("/home/ec2-user/workspace/pv-core/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon
from plotReconError import plotReconError

#For plotting
#import matplotlib.pyplot as plt

outputDir = "/home/ec2-user/mountData/benchmark/train/rcorr/aws_icaweights_binoc_RELU/"
skipFrames = 1 #Only print every 20th frame
startFrames = 0
doPlotRecon = True
doPlotErr = False
errShowPlots = False
layers = [
   "a1_LeftRecon",
   "a2_RightRecon",
   "a4_DepthDownsample"
   ]

#Layers for constructing recon error
preErrLayers = [
   "a1_LeftDownsample",
   "a5_RightDownsample",
]

postErrLayers = [
   "a3_LeftRecon",
   "a7_RightRecon",
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
   .007,
   .007,
]


if(doPlotRecon):
   print("Plotting reconstructions")
   plotRecon(layers, outputDir, skipFrames)

if(doPlotErr):
   print("Plotting reconstruction error")
   plotReconError(preErrLayers, postErrLayers, preToPostScale, outputDir, errShowPlots, skipFrames, gtLayers)
