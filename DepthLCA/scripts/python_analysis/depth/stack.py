import os, sys
lib_path = os.path.abspath("/home/slundquist/workspace/PetaVision/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon

#For plotting
#import matplotlib.pyplot as plt

outputDir = "/nh/compneuro/Data/Depth/LCA/depth_imprint_stack/"
skipFrames = 100 #Only print every 20th frame
doPlotRecon = True
doPlotErr = False
errShowPlots = False
layers = [
   "a3_LeftRescale",
   "a5_LeftRecon",
   "a10_LeftDepthRescale",
   "a12_LeftDepthRecon",
   "a16_RightRescale",
   "a18_RightRecon",
   "a23_RightDepthRescale",
   "a25_RightDepthRecon",
   "a28_PosRescale",
   "a30_PosRecon"
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
