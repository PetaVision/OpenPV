import os, sys
lib_path = os.path.abspath("/home/ec2-user/workspace/pv-core/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon
from plotReconError import plotReconError

#For plotting
#import matplotlib.pyplot as plt

outputDir = "/home/ec2-user/mountData/MAX/deconvolveS2/";
skipFrames = 1 #Only print every 20th frame
startFrames = 0
doPlotRecon = True
layers = [
   "a3_ImageDeconS2",
   ]

if(doPlotRecon):
   print("Plotting reconstructions")
   plotRecon(layers, outputDir, skipFrames)

