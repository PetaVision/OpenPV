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
import matplotlib.pyplot as plt


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
l1_layer   = 'a2_L1.pvp'
err_layer  = 'a1_Residual.pvp'
weights    = 'w1_L1_to_Residual.pvp'
#weights    = 'checkpoints/Checkpoint23400/L1_to_Residual_W.pvp'

# Open files
l1_activityFile  = open(output_dir + l1_layer,'rb')
err_activityFile = open(output_dir + err_layer,'rb')
weightsFile      = open(output_dir + weights,'rb')

progressPeriod  = 1
startFrame      = 0
lastFrame       = -1  # -1 for all
skipFrames      = 1

# outStruct has fields "time" and "values"

#print 'L1:'
#(L1_outStruct,L1_hdr)   = pv.get_pvp_data(l1_activityFile,progressPeriod,lastFrame,startFrame,skipFrames)

# divide the L2 norm of the residual by the L2 of the input to the
# residiual (i.e the image) to get % error
#print 'Err:'
#(err_outStruct,err_hdr) = pv.get_pvp_data(err_activityFile,progressPeriod,lastFrame,startFrame,skipFrames)

#Recon error?
# ABS gives distance from 0. 
# Averaging over the 512x512 array gives err per frame
#plt.plot(np.average(np.average(np.abs(err_outStruct["values"]),2),2))
#plt.show()

print 'Weights:'
(weightStruct,weightsHdr) = pv.get_pvp_data(weightsFile,progressPeriod,lastFrame,startFrame,skipFrames)

#l1_activityFile.close()
#err_activityFile.close()
weightsFile.close()


###########################################
###########################################
###########################################

margin = 6 #pixels

# weightStruct should be dims [time, numArbors, numPatches, nyp, nxp, nfp]
weight_vals = np.array(weightStruct["values"])
weight_time = weightStruct["time"]

(numFrames,numArbors,numPatches,nyp,nxp,nfp) = weight_vals.shape
i_frame = 400
i_arbor = 0

if np.sqrt(numPatches)%1 == 0: #If numPaches has a square root
    numPatchesX = np.sqrt(numPatches)
    numPatchesY = numPatchesX
else:
    numPatchesX = np.floor(np.sqrt(numPatches))
    numPatchesY = numPatchesX+1

patch_set = np.zeros((numPatchesX*numPatchesY,nyp+margin,nxp+margin))

for i_patch in range(numPatches): 
    patch_tmp = weight_vals[i_frame,i_arbor,i_patch,:,:,:]
    min_patch = np.amin(patch_tmp)
    max_patch = np.amax(patch_tmp)
    # Normalize patch
    patch_tmp = (patch_tmp - min_patch) * 255 / (max_patch - min_patch + np.finfo(float).eps) # re-scaling & normalizing TODO: why? and what exactly is it doing?
    # Patches are padded with zeros - just fill in center
    patch_set[i_patch,np.floor(margin/2):np.floor(margin/2)+nyp,np.floor(margin/2):np.floor(margin/2)+nxp] = np.uint8(np.squeeze(np.transpose(patch_tmp,(1,0,2)))) # re-ordering to [x,y,f] TODO: why?

#TODO: Does this actually reshape it how I want it to? (hint: no...)
out_mat = np.reshape(patch_set,(numPatchesY*(nyp+margin),numPatchesX*(nxp+margin)),'C')

plt.imshow(out_mat,cmap='Greys',interpolation='nearest')
plt.show()
