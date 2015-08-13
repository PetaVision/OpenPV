import os, sys
lib_path = os.path.abspath("/home/ec2-user/workspace/pv-core/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon
from plotReconError import plotReconError

#For plotting
#import matplotlib.pyplot as plt

outputDir = "/home/ec2-user/mountData/dictLearn/aws_icapatch/"
skipFrames = 1 #Only print every 20th frame
startFrames = 0
doPlotRecon = True
doPlotErr = False
errShowPlots = False
layers = [
   "a5_LeftRecon",
   "a11_RightRecon",
   "a3_LeftRescale",
   "a9_RightRescale",
   ]

#layers = [
#   "a4_LeftImageDecon",
#   "a5_RightImageDecon",
#   "a21_LeftImageDeconS3",
#   "a22_RightImageDeconS3",
#   "a15_LeftImageDeconS2",
#   "a16_RightImageDeconS2",
#   "a10_LeftImageDeconS1",
#   "a11_RightImageDeconS1",
#   "a0_LeftImage",
#   "a1_RightImage",
#   ]

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
