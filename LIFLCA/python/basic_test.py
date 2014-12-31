###############################
##  LCA ANALYSIS
##  Dylan Paiton
##  Will Shainin
##
## TODO:
## Sparse Layers
##    - Percent Active
##    - Percent Change?
##    - Activity Histogram
## Non-Sparse Layers
##    - Recon/Input Images
##    - Error
## Weight File
##    - Plot Weights
##
###############################

#workspace_path = '/home/ec2-user/mountData/'
workspace_path = '/Users/dpaiton/Documents/workspace/'

import os, sys
#lib_path = os.path.abspath('/home/ec2-user/workspace/PetaVision/plab')
lib_path = os.path.abspath(workspace_path+'PetaVision/plab')
sys.path.append(lib_path)
import pvAnalysis as pv
import plotWeights as pw
import plotError as pe
import numpy as np
import matplotlib.pyplot as plt
import math as math

# File Locations
output_dir   = workspace_path+'LIFLCA/output/LCA/'
input_layer  = 'a0_Input.pvp'
l1_layer     = 'a2_L1.pvp'
err_layer    = 'a1_Residual.pvp'
recon_layer  = 'a3_Recon.pvp'
weights      = 'w1_L1_to_Residual.pvp'
weights_chk  = 'checkpoints/Checkpoint400/L1_to_Residual_W.pvp'

# Open files
input_activityFile  = open(output_dir + input_layer,'rb')
l1_activityFile     = open(output_dir + l1_layer,'rb')
err_activityFile    = open(output_dir + err_layer,'rb')
recon_activityFile  = open(output_dir + recon_layer,'rb')
weightsFile         = open(output_dir + weights,'rb')
weightsChkFile      = open(output_dir + weights_chk,'rb')

progressPeriod  = 1
startFrame      = 0
lastFrame       = -1  # -1 for all
skipFrames      = 1   # 1 is every frame


########################
## READ IN FILES
########################
# outDat has fields "time" and "values"

#size is (numFrames,ny,nx,nf)
print('Input:')
(inputDat,inputHdr)   = pv.get_pvp_data(input_activityFile,progressPeriod,lastFrame,startFrame,skipFrames)
#frame = 0
#plt.imshow(np.squeeze(inputDat["values"][frame]),cmap='gray')
#plt.show(block=False)

#size is (numFrames,ny,nx,nf)
#print('L1:')
#(L1Dat,L1Hdr)   = pv.get_pvp_data(l1_activityFile,progressPeriod,lastFrame,startFrame,skipFrames)

#print('Err_from_file:')
#(errDat,errHdr)   = pv.get_pvp_data(err_activityFile,progressPeriod,lastFrame,startFrame,skipFrames)

#print('Err_from_recon:')
#(errDat,errHdr) = pv.get_pvp_data(err_activityFile,progressPeriod,lastFrame,startFrame,skipFrames)

#print('Weights from non-checkpoint:')
#(weightDat,weightsHdr) = pv.get_pvp_data(weightsFile,progressPeriod,lastFrame,startFrame,skipFrames)

print('Recon:')
(reconDat,reconHdr)   = pv.get_pvp_data(recon_activityFile,progressPeriod,lastFrame,startFrame,skipFrames)


########################
## ACTIVITY PLOTS
########################


########################
## RECON ERROR
########################
#TODO: verify that SNRdb and percErr are as expected
showPlot = True
savePlot = False
SNRdb   = pe.plotSnrDbErr(inputDat,reconDat,showPlot,savePlot,'./SNRTest.png')
percErr = pe.plotPercErr(inputDat,reconDat,showPlot,savePlot,'./percTest.png')


########################
## PLOT WEIGHTS
########################
#input_activityFile.close()
#l1_activityFile.close()
#err_activityFile.close()
#recon_activityFile.close()
#weightsFile.close()
#weightsChkFile.close()
#
#i_arbor    = 0
#i_frame    = 0 # index, not actual frame number
#margin     = 2 #pixels
#showPlot   = True
#savePlot   = False
#saveName   = output_dir+'analysis/'+weights[:-4]+'_'+str(i_frame).zfill(5)+'.png'
#
#weight_mat = pw.plotWeights(weightDat,i_arbor,i_frame,margin,showPlot,savePlot,saveName)

