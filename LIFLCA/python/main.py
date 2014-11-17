###############################
##  LCA ANALYSIS
##  Dylan Paiton
##  
###############################

import os, sys
lib_path = os.path.abspath('/Users/dpaiton/Documents/workspace/PetaVision/plab')
sys.path.append(lib_path)
import pvAnalysis as pv
import numpy as np


# Problem 1 setup:
# 
#   585 512x512 grayscale images, presented sequentially
#   8x8 image patches, evenly tiled across image
#   1 image per batch
#   256 dictionary elements - 2x overcomplete because of rectification
#   L-1 gradient descent
#   
#   displayPeriod = 40ms
#   timeConstantTau = 100
#   VThresh = 0.05
#   dWMax = 1.0


# File Locations
output_dir = '/Users/dpaiton/Documents/workspace/LIFLCA/output/LCA/'
layer   = 'a2_L1.pvp'
weights = 'checkpoints/Checkpoint23400/L1_to_Residual_W.pvp'

# Open file
activityFile = open(output_dir + layer,'rb')

progressPeriod  = 1
startFrame      = 0
lastFrame       = 3
skipFrames      = 1

(outStruct,hdr) = pv.get_pvp_data(activityFile,progressPeriod,lastFrame,startFrame,skipFrames)

