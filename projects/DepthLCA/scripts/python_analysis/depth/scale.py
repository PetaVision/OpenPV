import os, sys
lib_path = os.path.abspath("/home/slundquist/workspace/PetaVision/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon
from plotReconError import plotReconError

#For plotting
#import matplotlib.pyplot as plt

outputDir = "/nh/compneuro/Data/Depth/LCA/depth_log_scale/"
skipFrames = 20 #Only print every 20th frame
doPlotRecon = False
doPlotErr = True
errShowPlots = False
layers = [
   "a10_LeftRescale1",
   "a11_LeftRescale2",
   "a12_LeftRescale3",
   "a16_LeftRecon1",
   "a17_LeftRecon2",
   "a18_LeftRecon3",
   "a23_LeftDepthRescale",
   "a25_LeftDepthRecon",
   "a36_RightRescale1",
   "a37_RightRescale2",
   "a38_RightRescale3",
   "a42_RightRecon1",
   "a43_RightRecon2",
   "a44_RightRecon3",
   "a49_RightDepthRescale",
   "a51_RightDepthRecon",
   "a54_PosRescale",
   "a56_PosRecon"
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
   plotReconError(preErrLayers, postErrLayers, preToPostScale, outputDir, errShowPlots, skipFrames)
