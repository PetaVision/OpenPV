import os, sys
lib_path = os.path.abspath("/home/slundquist/workspace/PetaVision/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon

#For plotting
#import matplotlib.pyplot as plt

outputDir = "/nh/compneuro/Data/Depth/LCA/scale_v2/"
skipFrames = 1 #Only print every 20th frame
doPlotRecon = True
doPlotErr = False
errShowPlots = False
layers = [
   "a10_LeftRescale1",
   "a11_LeftRescale2",
   "a12_LeftRescale3",
   "a16_LeftRecon1",
   "a17_LeftRecon2",
   "a18_LeftRecon3",
   "a29_RightRescale1",
   "a30_RightRescale2",
   "a31_RightRescale3",
   "a35_RightRecon1",
   "a36_RightRecon2",
   "a37_RightRecon3",
   "a45_DepthRescale",
   "a47_DepthRecon",
   "a54_V2_1LeftRecon1",
   "a55_V2_1LeftRecon2",
   "a56_V2_1LeftRecon3",
   "a57_V2_1RightRecon1",
   "a58_V2_1RightRecon2",
   "a59_V2_1RightRecon3",
   ]
#Layers for constructing recon error
preErrLayers = [
   "a10_LeftRescale1",
   "a11_LeftRescale2",
   "a12_LeftRescale3",
   "a36_RightRescale1",
   "a37_RightRescale2",
   "a38_RightRescale3",
]
postErrLayers = [
   "a16_LeftRecon1",
   "a17_LeftRecon2",
   "a18_LeftRecon3",
   "a42_RightRecon1",
   "a43_RightRecon2",
   "a44_RightRecon3",
]

preToPostScale = [
   .0624,
   .0624,
   .0588,
   .0624,
   .0624,
   .0588,
]


if(doPlotRecon):
   print "Plotting reconstructions"
   plotRecon(layers, outputDir, skipFrames)

if(doPlotErr):
   print "Plotting reconstruction error"
   plotReconError(preErrLayers, postErrLayers, preToPostScale, outputDir, errShowPlots)
