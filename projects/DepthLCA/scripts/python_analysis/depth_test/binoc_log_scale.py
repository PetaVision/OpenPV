import os, sys
lib_path = os.path.abspath("/home/slundquist/workspace/PetaVision/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon
from plotReconError import plotReconError

outputDir = "/nh/compneuro/Data/Depth/LCA/depth_log_scale_test/"
skipFrames = 1 #Only print every 20th frame
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
   "a24_LeftDepthRecon",
   "a35_RightRescale1",
   "a36_RightRescale2",
   "a37_RightRescale3",
   "a41_RightRecon1",
   "a42_RightRecon2",
   "a43_RightRecon3",
   "a48_RightDepthRescale",
   "a49_RightDepthRecon",
   "a52_PosRescale",
   "a54_PosRecon"
   ]

if(doPlotRecon):
   print "Plotting reconstructions"
   plotRecon(layers, outputDir, skipFrames)
