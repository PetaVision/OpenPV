import os, sys
import pdb
lib_path = os.path.abspath("/home/ec2-user/workspace/pv-core/plab/")
sys.path.append(lib_path)
from plotRecon import plotRecon
from plotReconError import plotReconError

#For plotting
#import matplotlib.pyplot as plt

baseDir = "/home/ec2-user/mountData/benchmark/featuremap/norect/icaweights_LCA/"

tuningFile = "/home/ec2-user/mountData/benchmark/featuremap/norect/LCA_peakmean.txt"

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

for i, r in enumerate(rank):
   dataDir = baseDir + "paramsweep_" + str(r-1).zfill(3) + "/"
   plotRecon(layers, dataDir, skipFrames, 0, True, baseDir, "_rank"+str(i+1).zfill(3)+"_neuron"+str(r).zfill(3))


baseDir = "/home/ec2-user/mountData/benchmark/featuremap/norect/icaweights_LCA_all/"
##One recon for all combined
plotRecon(layers, baseDir, skipFrames, 0, True, baseDir, "_all")


