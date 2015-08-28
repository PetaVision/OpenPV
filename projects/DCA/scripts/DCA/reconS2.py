import os, sys
lib_path = os.path.abspath("/home/slundquist/workspace/OpenPV/pv-core/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon
from plotReconError import plotReconError

#For plotting
#import matplotlib.pyplot as plt

outputDir = "/nh/compneuro/Data/DCA/deconvolveS2/"
skipFrames = 1 #Only print every 20th frame
startFrames = 0
doPlotRecon = True
layers = [
   "a2_ImageDeconS2",
   ]

if(doPlotRecon):
   print("Plotting reconstructions")
   plotRecon(layers, outputDir, skipFrames)

