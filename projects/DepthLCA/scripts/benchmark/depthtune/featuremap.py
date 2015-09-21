import os, sys
import pdb
lib_path = os.path.abspath("/home/ec2-user/workspace/pv-core/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon
from plotReconError import plotReconError

#For plotting
#import matplotlib.pyplot as plt

baseDir = "/home/ec2-user/mountData/benchmark/featuremap/icaweights_LCA/"
dataDirs = [baseDir + "paramsweep_" + str(i).zfill(3) + "/" for i in range(0, 512)]

tuningFile = "/home/ec2-user/mountData/benchmark/featuremap/LCA_peakmean.txt"

tf = open(tuningFile, 'r')

tfLines = tf.readlines()
tf.close()

#Remove first line
tfLines.pop(0)

rank = [int(line.split(":")[0]) for line in tfLines]

skipFrames = 1 #Only print every 20th frame
startFrames = 0
doPlotRecon = True

layers = [
   "LeftRecon_slice",
   "RightRecon_slice"
   ]


print("Plotting reconstructions")

#for i, dataDir in enumerate(dataDirs):
#   suffixIdx = rank[i];
#   plotRecon(layers, dataDir, skipFrames, 0, True, baseDir, "_"+str(suffixIdx))

#One recon for all combined
plotRecon(layers, baseDir, skipFrames, 0, True, baseDir, "_all")


