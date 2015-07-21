import os, sys
lib_path = os.path.abspath("/home/slundquist/workspace/PetaVision/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon
from plotReconError import plotReconError

#For plotting
#import matplotlib.pyplot as plt

outputDir = "/nh/compneuro/Data/Depth/LCA/stack_mono/"
skipFrames = 20 #Only print every 20th frame
doPlotRecon = True
doPlotErr = True
errShowPlots = False
layers = [
      "a3_LeftRescale",
      "a5_LeftReconS2",
      "a6_LeftReconS4",
      "a7_LeftReconS8",
      "a8_LeftReconAll",
      #"a12_RightRescale",
      #"a14_RightReconS2",
      #"a15_RightReconS4",
      #"a16_RightReconS8",
      #"a17_RightReconAll",
   ]
#Layers for constructing recon error

preErrLayers = [
   "a8_LeftReconAll",
   "a8_LeftReconAll",
   "a8_LeftReconAll",
   #"a17_RightReconAll",
   #"a17_RightReconAll",
   #"a17_RightReconAll",

   "a3_LeftRescale",
   "a3_LeftRescale",
   "a3_LeftRescale",
   #"a12_RightRescale",
   #"a12_RightRescale",
   #"a12_RightRescale",
]
postErrLayers = [
   "a5_LeftReconS2",
   "a6_LeftReconS4",
   "a7_LeftReconS8",
   #"a14_RightReconS2",
   #"a15_RightReconS4",
   #"a16_RightReconS8",

   "a5_LeftReconS2",
   "a6_LeftReconS4",
   "a7_LeftReconS8",
   #"a14_RightReconS2",
   #"a15_RightReconS4",
   #"a16_RightReconS8",
]

#TODO not used anymore
normalizeIdx = [-1, 0, 0, 0]

preToPostScale = [
   1,
   1,
   1,
   #1,
   #1,
   #1,

   .0294,
   .0294,
   .0294,
   #.0294,
   #.0294,
   #.0294,
]


if(doPlotRecon):
   print "Plotting reconstructions"
   plotRecon(layers, outputDir, skipFrames)

if(doPlotErr):
   print "Plotting reconstruction error"
   plotReconError(preErrLayers, postErrLayers, preToPostScale, outputDir, errShowPlots, normalizeIdx, skipFrames)
