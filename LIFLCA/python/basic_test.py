###############################
##  LCA ANALYSIS
##  Dylan Paiton
##  Will Shainin
###############################

#workspace_path = '/home/wshainin/workspace/'
workspace_path = '/Users/dpaiton/Documents/workspace/'

import os, sys
lib_path = os.path.abspath(workspace_path+'PetaVision/plab')
sys.path.append(lib_path)
import pvAnalysis as pv
import plotWeights as pw
import numpy as np
import matplotlib.pyplot as plt

# File Locations
#output_dir   = workspace_path+'/awsMount/mountData/LIFLCA/output/LCA/'
output_dir   = workspace_path+'/LIFLCA/output_small/LCA/'
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

# outStruct has fields "time" and "values"

#size is (numFrames,nf,nx,ny)
#print('Input:')
#(inputStruct,inputHdr)   = pv.get_pvp_data(input_activityFile,progressPeriod,lastFrame,startFrame,skipFrames)
# np.array(InputStruct["values"]).shape
# (361, 1, 512, 512)
# has 361 "frames", which I guess makes sense if initialWriteTime = 40 (400-39 = 361)
# why is it 512x512?

print('L1:')
(L1Struct,L1Hdr)   = pv.get_pvp_data(l1_activityFile,progressPeriod,lastFrame,startFrame,skipFrames)
# takes a long time but runs without error
# 10 frames total
# np.array(L1Struct["values]).shape
# (10, 32, 32, 256)

#print('Err_from_file:')
#(errStruct,errHdr)   = pv.get_pvp_data(err_activityFile,progressPeriod,lastFrame,startFrame,skipFrames)

#print('Recon:')
#(reconStruct,reconHdr)   = pv.get_pvp_data(recon_activityFile,progressPeriod,lastFrame,startFrame,skipFrames)

# Gar Method
# divide the L2 norm of the residual by the L2 of the input to the
# residiual (i.e the image) to get % error
#TODO: pass param for error method, there are 3 that I know of. Gar method, pSNR, SNR
#print('Err_from_recon:')
#(errStruct,errHdr) = pv.get_pvp_data(err_activityFile,progressPeriod,lastFrame,startFrame,skipFrames)

#print('Weights from non-checkpoint:')
#(weightStruct,weightsHdr) = pv.get_pvp_data(weightsFile,progressPeriod,lastFrame,startFrame,skipFrames)

#print('Weights from checkpoint:')
#(weightChkStruct,weightsChkHdr) = pv.get_pvp_data(weightsChkFile,progressPeriod,lastFrame,startFrame,skipFrames)

#input_activityFile.close()
#l1_activityFile.close()
#err_activityFile.close()
#recon_activityFile.close()
#weightsFile.close()
#weightsChkFile.close()

#i_arbor    = 0
#i_frame    = 400 # index, not actual frame number
#margin     = 2 #pixels
#showPlot   = True
#savePlot   = True
#saveName   = output_dir+'analysis/'+weights[:-4]+'_'+str(i_frame).zfill(5)+'.png'
#
#weight_mat = pw.plotWeights(weightStruct,i_arbor,i_frame,margin,showPlot,savePlot,saveName)
