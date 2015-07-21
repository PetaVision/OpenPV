import os, sys
lib_path = os.path.abspath("/home/slundquist/workspace/PetaVision/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon
from plotReconError import plotReconError

#For plotting
#import matplotlib.pyplot as plt

outputDir = "/nh/compneuro/Data/MLPLCA/LCA/cifar_classvis/"
skipFrames = 1 #Only print every 20th frame
doPlotRecon = True
doPlotErr = False
errShowPlots = False
layers = [
   "0/a3_Recon",
   "1/a3_Recon",
   "2/a3_Recon",
   "3/a3_Recon",
   "4/a3_Recon",
   "5/a3_Recon",
   "6/a3_Recon",
   "7/a3_Recon",
   "8/a3_Recon",
   "9/a3_Recon",
   ]

if(doPlotRecon):
   print "Plotting reconstructions"
   plotRecon(layers, outputDir, skipFrames)

#if(doPlotErr):
#   print "Plotting reconstruction error"
#   plotReconError(preErrLayers, postErrLayers, preToPostScale, outputDir, errShowPlots, skipFrames, gtLayers)
