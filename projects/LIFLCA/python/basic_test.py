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

workspace_path = '/Users/dpaiton/Documents/workspace/'

import os, sys
lib_path = os.path.abspath(workspace_path+'PetaVision/plab')
sys.path.append(lib_path)
import pvAnalysis as pv
import plotActivity as pa
import plotWeights as pw
import plotError as pe
import numpy as np
import matplotlib.pyplot as plt
import math as math

# File Locations
output_dir   = workspace_path+'LIFLCA/output/heli_LCA/'
#output_dir   = './'
input_layer  = 'a0_Input.pvp'
#l1_layer     = 'a2_L1.pvp'
l1_layer     = 'checkpoints/Checkpoint20000/L1_A.pvp'
err_layer    = 'a1_Residual.pvp'
recon_layer  = 'a3_Recon.pvp'
weights      = 'w1_L1_to_Residual.pvp'
weights_chk  = 'checkpoints/Checkpoint20000/L1_to_Residual_W.pvp'

# Open files
#input_activityFile  = open(output_dir + input_layer,'rb')
l1_activityFile     = open(output_dir + l1_layer,'rb')
#err_activityFile    = open(output_dir + err_layer,'rb')
#recon_activityFile  = open(output_dir + recon_layer,'rb')
#weightsFile         = open(output_dir + weights,'rb')
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
#print('Input:')
#(inputDat,inputHdr)   = pv.get_pvp_data(input_activityFile,progressPeriod,lastFrame,startFrame,skipFrames)
#frame = 0
#plt.imshow(np.squeeze(inputDat["values"][frame]),cmap='gray')
#plt.show(block=False)

#size is (numFrames,ny,nx,nf)
print('L1:')
(L1Dat,L1Hdr)   = pv.get_pvp_data(l1_activityFile,progressPeriod,lastFrame,startFrame,skipFrames)

#print('Err_from_file:')
#(errDat,errHdr)   = pv.get_pvp_data(err_activityFile,progressPeriod,lastFrame,startFrame,skipFrames)

#print('Err_from_recon:')
#(errDat,errHdr) = pv.get_pvp_data(err_activityFile,progressPeriod,lastFrame,startFrame,skipFrames)

#print('Weights from non-checkpoint:')
#(weightDat,weightsHdr) = pv.get_pvp_data(weightsFile,progressPeriod,lastFrame,startFrame,skipFrames)

print('Weights from checkpoint:')
(weightChkDat,weightsChkHdr) = pv.get_pvp_data(weightsChkFile,progressPeriod,lastFrame,startFrame,skipFrames)

#print('Recon:')
#(reconDat,reconHdr)   = pv.get_pvp_data(recon_activityFile,progressPeriod,lastFrame,startFrame,skipFrames)


########################
## ACTIVITY PLOTS
########################
#percActive = pa.plotPercentActive(L1Dat,showPlot=True,savePlot=False,saveName='')
#percChange = pa.plotPercentChange(L1Dat,showPlot=True,savePlot=False,saveName='')


########################
## RECON ERROR
########################
#TODO: verify that SNRdb and percErr are as expected
#showPlot = True
#savePlot = False
#SNRdb   = pe.plotSnrDbErr(inputDat,reconDat,showPlot,savePlot,'./SNRTest.png')
#percErr = pe.plotPercErr(inputDat,reconDat,showPlot,savePlot,'./percTest.png')


########################
## PLOT WEIGHTS
########################
#L1Dat      = None
#arborIdx   = np.arange(0,4)
arborIdx   = None
i_frame    = -1 # index, not actual frame number, -1 for last
margin     = 2 #pixels
plotColor  = True 
showPlot   = False 
savePlot   = True 
saveName   = output_dir+'analysis/'+weights_chk[:-4]+'.png'

weight_list = pw.plotWeights(weightChkDat,L1Dat,arborIdx,i_frame,margin,plotColor,showPlot,savePlot,saveName)

#plotColor  = True
#showPlot   = True 
#savePlot   = True 
#saveName   = output_dir+'analysis/'+weights_chk[:-4]+'_sorted.png'
#
#sorted_weight_list = pw.plotSortedWeights(weight_list,L1Dat,plotColor,showPlot,savePlot,saveName)
#

########################
## CLOSE FILESTREAMS
########################
#input_activityFile.close()
#l1_activityFile.close()
#err_activityFile.close()
#recon_activityFile.close()
#weightsFile.close()
weightsChkFile.close()
