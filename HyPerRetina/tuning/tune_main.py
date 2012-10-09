#!/usr/bin/env python

#For tuning Gerd Kunde & Gar Kenyon's retina
#Created by Dylan Paiton

import os
import re
from tune_functions import *

###################
###             ###
### Global vars ###
###             ###
###################

run_PetaVision      = 1       #Will just create params file if set to 0
pipe_output_to_file = 1
mpi_np              = '4'
mpi_rows            = '2'
mpi_columns         = '2'

num_steps_list      = ['14420']
stochastic_flag     = '1' #preActivityNotRate = !stochastic_flag

param_template_name = 'retina_params.template'
run_name            = 'AntiTerrorism'

#wrkspc_path         = '/Users/garkenyon/workspace-sync-anterior'
wrkspc_path         = '/Users/dpaiton/Documents/Work/LANL/workspace'
out_filename        = run_name
results_path        = wrkspc_path+'/HyPerRetina/output/'+run_name

input_path          = wrkspc_path+'/HyPerRetina/input'
tuning_path         = wrkspc_path+'/HyPerRetina/tuning'
param_in_file       = tuning_path+'/'+param_template_name
param_out_file      = tuning_path+'/params_files/'+out_filename
run_path            = wrkspc_path+'/HyPerRetina/Debug/HyPerRetina'

## INPUT FILE (One should be uncommented)
#input_image = input_path+'/amoeba/1f/sigma1/amoeba_1f_1_64_same_gjk.png'
#input_image = input_path+'/amoeba/1f/sigma20/background_1f_20_gjk.png'
#input_image = input_path+'/amoeba/1f/sigma1/background_1f_1_gjk.png'
#input_image = input_path+'/000396.png'
#input_image = input_path+'/blackimage.png'
input_image = input_path+'/gray128image.png'

## INPUT MOVIE (One should be uncommented)
#input_movie = input_path+'filenamesnjitter.txt';
#input_movie = input_path+'/filenames_graywhiteblack.txt'
#input_movie = input_path+'/filenames_graywhiteblackspots.txt'
input_movie = input_path+'/filenames_movie.txt'

## Declare layers
#INPUTS
Image                = 0 
Movie                = 1
Patterns             = 0

#ANN INPUT COPY
ImageBuffer          = 1
ConstantVrest        = 1

#CONE
Cone                 = 1
ConeSigmoidON        = 1
ConeSigmoidOFF       = ConeSigmoidON

#BIPOLAR
BipolarON            = 1
BipolarSigmoidON     = 1
BipolarOFF           = BipolarON
BipolarSigmoidOFF    = BipolarSigmoidON

#HORIZONTAL
Horizontal           = 1
HoriGap              = 1
HoriSigmoid          = 1

#WFAMACRINE
WFAmacrineON         = 1
WFAmacrineSigmoidON  = 1
WFAmacrineOFF        = WFAmacrineON
WFAmacrineSigmoidOFF = WFAmacrineSigmoidON

#PAAmacrie
PAAmacrineON         = 1
PAAmaGapON           = 1
PAAmacrineOFF        = PAAmacrineON
PAAmaGapOFF          = PAAmaGapON

#SFAmacrine
SFAmacrine           = 1
SFAmacrineSigmoid    = 1
SFAmacrineGap        = 1

#GANGLION
GanglionON           = 1
GangliGapON          = 1
GanglionOFF          = GanglionON
GangliGapOFF         = GangliGapON

## Declare conn strength values
##    frange is (start, stop, step)
ImageImageBuffer             = ["%g" % x for x in frange(40,0,0)]   # Image to ImageBuffer
ConstantVrestImageBuffer     = ["%g" % x for x in frange(1,0,0)]    # ConstantVrest to ImageBuffer
ImageBufferCone              = ["%g" % x for x in frange(1,0,0)]    # ImageBuffer to Cone

ConeSigmoidBipolar           = ["%g" % x for x in frange(0.5,0,0)]  # ConeSigmoid to Bipolar
ConeSigmoidHorizontal        = ["%g" % x for x in frange(0.5,0,0)]  # ConeSigmoid to Horizontal

HoriGapHorizontal            = ["%g" % x for x in frange(3,0,0)]    # HoriGap to Horizontal
HoriSigmoidCone              = ["%g" % x for x in frange(1,0,0)]    # HoriSigmoid to Cone

BipolarSigmoidWFAmacrine     = ["%g" % x for x in frange(1,0,0)]    # BipolarSigmoid to WFAmacrine
BipolarSigmoidGanglion       = ["%g" % x for x in frange(2,0,0)]    # BipolarSigmoid to Ganglion
BipolarSigmoidSFAmacrine     = ["%g" % x for x in frange(1,0,0)]    # BipolarSigmoid to SFAmacrine

WFAmacrineSigmoidBipolarON   = ["%g" % x for x in frange(0.15,0,0)]  # WFAmacrineSigmoidON to Bipolar
WFAmacrineSigmoidBipolarOFF  = ["%g" % x for x in frange(0.25,0,0)] # WFAmacrineSigmoidOFF to Bipolar
WFAmacrineONSFAmacrine       = ["%g" % x for x in frange(2,0,0)]    # WFAmacrineON to SFAmacrine
WFAmacrineOFFSFAmacrine      = ["%g" % x for x in frange(2,0,0)]    # WFAmacrineOFF to SFAmacrine

SFAmacrineGapSFAmacrine      = ["%g" % x for x in frange(1,0,0)]    # SFAmacrineGAP to SFAmacrine
SFAmacrineSigmoidGanglionON  = ["%g" % x for x in frange(0,0,0)]    # SFAmacrineSigmoid to GanglionON
SFAmacrineSigmoidGanglionOFF = ["%g" % x for x in frange(0,0,0)]    # SFAmacrineSigmoid to GanglionOFF
SFAmacrineSigmoidPAAmacrine  = ["%g" % x for x in frange(2,0,0)]    #Doubled bc we cut the tau # SFAmacrineSigmoid to PAAmacrine

PAAmaGapGanglion             = ["%g" % x for x in frange(2,0,0)]    # PAAmaGap to Ganglion
PAAmaGapPAAmacrine           = ["%g" % x for x in frange(1,0,0)]    # PAAmaGap to PAAmacrine
PAAmacrineGanglion           = ["%g" % x for x in frange(2,0,0)]    #Doubled bc we halved the Tinh# PAAmacrine to Ganglion
PAAmacrineWFAmacrine         = ["%g" % x for x in frange(1,0,0)]    # PAAmacrine to WFAmacrine
PAAmacrinePAAmacrine         = ["%g" % x for x in frange(2,0,0)]    #Doubled bc we cut the tau # PAAmacrine to PAAmacrine

GangliGapPAAmacrine          = ["%g" % x for x in frange(2,0,0)]    # GangliGap to PAAmacrine

#List possible connections
conn_list = ["ImageImageBuffer",
            "ConstantVrestImageBuffer",
            "ImageBufferCone",
            "ConeSigmoidBipolar",
            "ConeSigmoidHorizontal",
            "HoriGapHorizontal",
            "HoriSigmoidCone",
            "BipolarSigmoidWFAmacrine",
            "SFAmacrineGapSFAmacrine",
            "WFAmacrineSigmoidBipolarON",
            "WFAmacrineSigmoidBipolarOFF",
            "BipolarSigmoidGanglion",
            "GangliGapPAAmacrine",
            "PAAmaGapGanglion",
            "PAAmaGapPAAmacrine",
            "PAAmacrineGanglion",
            "PAAmacrinePAAmacrine",
            "BipolarSigmoidSFAmacrine",
            "SFAmacrineSigmoidGanglionON",
            "SFAmacrineSigmoidGanglionOFF",
            "SFAmacrineSigmoidPAAmacrine",
            "PAAmacrineWFAmacrine",
            "WFAmacrineONSFAmacrine",
            "WFAmacrineOFFSFAmacrine"]

conn_lol = [ImageImageBuffer,
            ConstantVrestImageBuffer,
            ImageBufferCone,
            ConeSigmoidBipolar,
            ConeSigmoidHorizontal,
            HoriGapHorizontal,
            HoriSigmoidCone,
            BipolarSigmoidWFAmacrine,
            SFAmacrineGapSFAmacrine,
            WFAmacrineSigmoidBipolarON,
            WFAmacrineSigmoidBipolarOFF,
            BipolarSigmoidGanglion,
            GangliGapPAAmacrine,
            PAAmaGapGanglion,
            PAAmaGapPAAmacrine,
            PAAmacrineGanglion,
            PAAmacrinePAAmacrine,
            BipolarSigmoidSFAmacrine,
            SFAmacrineSigmoidGanglionON,
            SFAmacrineSigmoidGanglionOFF,
            SFAmacrineSigmoidPAAmacrine,
            PAAmacrineWFAmacrine,
            WFAmacrineONSFAmacrine,
            WFAmacrineOFFSFAmacrine]

print "tune_params: Verifying parameters."
## Assert that all parameter lists are the same length or of length 1
max_list_len = max([len(x) for x in conn_lol]) #max lenght of sub list in lol
if not all(len(i)==max_list_len or len(i)==1 for i in conn_lol):
    exit("\ntune_params: ERROR: One of the lists is not the right size!\n")

## Check to see if any of the strengths are set to 0
##   nonZeroStrength is true if there is a nonzero strength (false if strength is 0)
nonZeroStrength = [strength not in '0' for connlist in conn_lol for strength in [max(connlist)]] # max val in each list is not 0
if len(conn_lol) is not len(nonZeroStrength):
    exit("\ntune_params: ERROR: nonZeroStrength array is not the appropriate length")

## Open file
if os.path.isfile(param_in_file):
    try:
        print "tune_params: Opening template param file "+param_in_file+"."
        in_fid = open(param_in_file)
        param_lines = in_fid.readlines()
        in_fid.close()
    except IOError as e:
        print "tune_params: Failed to open file "+param_in_file+" with error:\n"
        exit(e)
else:
    exit("\ntune_params: ERROR: Couldn't find file "+param_in_file+"!\n")

## Modify pvp file and run petavision for each parameter
for num_steps in num_steps_list:
    for param_idx in range(max_list_len):
        out_lines = param_lines[:] # Dereference to make copy of list

        idx_out_filename  = out_filename+str(param_idx)+'.pv'
        full_out_file     = param_out_file+'_p'+str(param_idx)+'_ns'+num_steps+'.pv'
        full_results_path = results_path+'/p'+str(param_idx)+'/ns'+num_steps

        print "tune_params: Modifying template file."
        for line_num in range(len(out_lines)):
            line = out_lines[line_num]

            ## Activate layers that have been set in the global vars section
            uncomment = False 
            if 'Image "Image"' in line and Image==1:                                            #########LAYERS
                uncomment = True 
            elif 'Movie "Image"' in line and Movie==1:
                uncomment = True 
            elif 'Patterns "Image"' in line and Patterns==1:
                uncomment = True 
            elif 'ANNLayer "ImageBuffer"' in line and ImageBuffer==1:
                uncomment = True
            elif 'ANNLayer "ConstantVrest"' in line and ConstantVrest==1:
                uncomment = True
            elif 'LIFGap "Cone"' in line and Cone==1:
                uncomment = True 
            elif 'SigmoidLayer "ConeSigmoidON"' in line and ConeSigmoidON==1:
                uncomment = True 
            elif 'SigmoidLayer "ConeSigmoidOFF"' in line and ConeSigmoidOFF==1:
                uncomment = True 
            elif 'LIF "BipolarON"' in line and BipolarON==1:
                uncomment = True 
            elif 'SigmoidLayer "BipolarSigmoidON"' in line and BipolarSigmoidON==1:
                uncomment = True 
            elif 'LIFGap "Horizontal"' in line and Horizontal==1:
                uncomment = True 
            elif 'GapLayer "HoriGap"' in line and HoriGap==1:
                uncomment = True
            elif 'SigmoidLayer "HoriSigmoid"' in line and HoriSigmoid==1:
                uncomment = True 
            elif 'LIF "WFAmacrineON"' in line and WFAmacrineON==1:
                uncomment = True 
            elif 'GapLayer "SFAmacrineGap"' in line and SFAmacrineGap==1:
                uncomment = True 
            elif 'SigmoidLayer "WFAmacrineSigmoidON"' in line and WFAmacrineSigmoidON==1:
                uncomment = True 
            elif 'LIFGap "GanglionON"' in line and GanglionON==1:
                uncomment = True 
            elif 'GapLayer "GangliGapON"' in line and GangliGapON==1:
                uncomment = True 
            elif 'LIFGap "PAAmacrineON"' in line and PAAmacrineON==1:
                uncomment = True 
            elif 'GapLayer "PAAmaGapON"' in line and PAAmaGapON==1:
                uncomment = True 
            elif 'LIF "BipolarOFF"' in line and BipolarOFF==1:
                uncomment = True
            elif 'SigmoidLayer "BipolarSigmoidOFF"' in line and BipolarSigmoidOFF==1:
                uncomment = True 
            elif 'LIF "WFAmacrineOFF"' in line and WFAmacrineOFF==1:
                uncomment = True 
            elif 'SigmoidLayer "WFAmacrineSigmoidOFF"' in line and WFAmacrineSigmoidOFF==1:
                uncomment = True 
            elif 'LIFGap "GanglionOFF"' in line and GanglionOFF==1:
                uncomment = True 
            elif 'GapLayer "GangliGapOFF"' in line and GangliGapOFF==1:
                uncomment = True 
            elif 'LIFGap "PAAmacrineOFF"' in line and PAAmacrineOFF==1:
                uncomment = True 
            elif 'GapLayer "PAAmaGapOFF"' in line and PAAmaGapOFF==1:
                uncomment = True
            elif 'LIFGap "SFAmacrine"' in line and SFAmacrine==1:
                uncomment = True
            elif 'SigmoidLayer "SFAmacrineSigmoid"' in line and SFAmacrineSigmoid==1:
                uncomment = True
            elif 'KernelConn "Image to ImageBuffer"' in line and ImageBuffer==1:                #########Connections
                zero_index = [idx for idx, enum in enumerate([param in 'ImageImageBuffer' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "ConstantVrest to ImageBuffer"' in line and ConstantVrest==1 and ImageBuffer==1:
                zero_index = [idx for idx, enum in enumerate([param in 'ConstantVrestImageBuffer' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'GapConn "ImageBuffer to Cone"' in line and Cone==1:
                zero_index = [idx for idx, enum in enumerate([param in 'ImageBufferCone' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "ConeSigmoidON to BipolarON"' in line and ConeSigmoidON==1 and BipolarON==1:
                zero_index = [idx for idx, enum in enumerate([param in 'ConeSigmoidBipolar' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "ConeSigmoidON to Horizontal"' in line and ConeSigmoidON==1 and Horizontal==1:
                zero_index = [idx for idx, enum in enumerate([param in 'ConeSigmoidHorizontal' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'GapConn "HoriGap to Horizontal"' in line and HoriGap==1 and Horizontal==1:
                zero_index = [idx for idx, enum in enumerate([param in 'HoriGapHorizontal' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "HoriSigmoid to Cone"' in line and HoriSigmoid==1 and Cone==1:
                zero_index = [idx for idx, enum in enumerate([param in 'HoriSigmoidCone' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "BipolarSigmoidON to GanglionON"' in line and BipolarSigmoidON==1 and GanglionON==1:
                zero_index = [idx for idx, enum in enumerate([param in 'BipolarSigmoidGanglion' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'GapConn "SFAmacrineGap to SFAmacrine"' in line and SFAmacrineGap==1 and SFAmacrine==1:
                zero_index = [idx for idx, enum in enumerate([param in 'SFAmacrineGapSFAmacrine' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "WFAmacrineSigmoidON to BipolarON"' in line and WFAmacrineSigmoidON==1 and BipolarON==1:
                zero_index = [idx for idx, enum in enumerate([param in 'WFAmacrineSigmoidBipolarON' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "BipolarSigmoidON to WFAmacrineON"' in line and BipolarSigmoidON==1 and WFAmacrineON==1:
                zero_index = [idx for idx, enum in enumerate([param in 'BipolarSigmoidWFAmacrine' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'GapConn "GangliGapON to PAAmacrineON"' in line and GangliGapON==1 and PAAmacrineON==1:
                zero_index = [idx for idx, enum in enumerate([param in 'GangliGapPAAmacrine' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'GapConn "PAAmaGapON to GanglionON"' in line and PAAmaGapON==1 and GanglionON==1:
                zero_index = [idx for idx, enum in enumerate([param in 'PAAmaGapGanglion' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'GapConn "PAAmaGapON to PAAmacrineON"' in line and PAAmaGapON==1 and PAAmacrineON==1:
                zero_index = [idx for idx, enum in enumerate([param in 'PAAmaGapPAAmacrine' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "PAAmacrineON to GanglionON"' in line and PAAmacrineON==1 and GanglionON==1:
                zero_index = [idx for idx, enum in enumerate([param in 'PAAmacrineGanglion' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "PAAmacrineON to PAAmacrineON"' in line and PAAmacrineON==1:
                zero_index = [idx for idx, enum in enumerate([param in 'PAAmacrinePAAmacrine' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "BipolarSigmoidON to SFAmacrine"' in line and BipolarSigmoidON==1 and SFAmacrine==1:
                zero_index = [idx for idx, enum in enumerate([param in 'BipolarSigmoidSFAmacrine' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "SFAmacrineSigmoid to GanglionON"' in line and SFAmacrineSigmoid==1 and GanglionON==1:
                zero_index = [idx for idx, enum in enumerate([param in 'SFAmacrineSigmoidGanglionON' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "SFAmacrineSigmoid to PAAmacrineON"' in line and SFAmacrineSigmoid==1 and PAAmacrineON==1:
                zero_index = [idx for idx, enum in enumerate([param in 'SFAmacrineSigmoidPAAmacrine' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "PAAmacrineON to WFAmacrineON"' in line and PAAmacrineON==1 and WFAmacrineON==1:
                zero_index = [idx for idx, enum in enumerate([param in 'PAAmacrineWFAmacrine' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "WFAmacrineSigmoidON to SFAmacrine"' in line and WFAmacrineSigmoidON==1 and SFAmacrine==1:
                zero_index = [idx for idx, enum in enumerate([param in 'WFAmacrineONSFAmacrine' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "WFAmacrineSigmoidOFF to SFAmacrine"' in line and WFAmacrineSigmoidOFF==1 and SFAmacrine==1:
                zero_index = [idx for idx, enum in enumerate([param in 'WFAmacrineOFFSFAmacrine' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "ConeSigmoidOFF to BipolarOFF"' in line and ConeSigmoidOFF==1 and BipolarOFF==1:
                zero_index = [idx for idx, enum in enumerate([param in 'ConeSigmoidBipolar' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "BipolarSigmoidOFF to WFAmacrineOFF"' in line and BipolarSigmoidOFF==1 and WFAmacrineOFF==1:
                zero_index = [idx for idx, enum in enumerate([param in 'BipolarSigmoidWFAmacrine' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "WFAmacrineSigmoidOFF to BipolarOFF"' in line and WFAmacrineSigmoidOFF==1 and BipolarOFF==1:
                zero_index = [idx for idx, enum in enumerate([param in 'WFAmacrineSigmoidBipolarOFF' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "BipolarSigmoidOFF to GanglionOFF"' in line and BipolarSigmoidOFF==1 and GanglionOFF==1:
                zero_index = [idx for idx, enum in enumerate([param in 'BipolarSigmoidGanglion' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "BipolarSigmoidOFF to SFAmacrine"' in line and BipolarSigmoidOFF==1 and SFAmacrine==1:
                zero_index = [idx for idx, enum in enumerate([param in 'BipolarSigmoidSFAmacrine' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'GapConn "GangliGapOFF to PAAmacrineOFF"' in line and GangliGapOFF==1 and PAAmacrineOFF==1:
                zero_index = [idx for idx, enum in enumerate([param in 'GangliGapPAAmacrine' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'GapConn "PAAmaGapOFF to GanglionOFF"' in line and PAAmaGapOFF==1 and GanglionOFF==1:
                zero_index = [idx for idx, enum in enumerate([param in 'PAAmaGapGanglion' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'GapConn "PAAmaGapOFF to PAAmacrineOFF"' in line and PAAmaGapOFF==1 and PAAmacrineOFF==1:
                zero_index = [idx for idx, enum in enumerate([param in 'PAAmaGapPAAmacrine' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "PAAmacrineOFF to GanglionOFF"' in line and PAAmacrineOFF==1 and GanglionOFF==1:
                zero_index = [idx for idx, enum in enumerate([param in 'PAAmacrineGanglion' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "PAAmacrineOFF to PAAmacrineOFF"' in line and PAAmacrineOFF==1:
                zero_index = [idx for idx, enum in enumerate([param in 'PAAmacrinePAAmacrine' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "SFAmacrineSigmoid to GanglionOFF"' in line and SFAmacrineSigmoid==1 and GanglionOFF==1:
                zero_index = [idx for idx, enum in enumerate([param in 'SFAmacrineSigmoidGanglionOFF' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "SFAmacrineSigmoid to PAAmacrineOFF"' in line and SFAmacrineSigmoid==1 and PAAmacrineOFF==1:
                zero_index = [idx for idx, enum in enumerate([param in 'SFAmacrineSigmoidPAAmacrine' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "PAAmacrineOFF to WFAmacrineOFF"' in line and PAAmacrineOFF==1 and WFAmacrineOFF==1:
                zero_index = [idx for idx, enum in enumerate([param in 'PAAmacrineWFAmacrine' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True
            elif 'KernelConn "WFAmacrineSigmoidOFF to SFAmacrine"' in line and WFAmacrineSigmoidOFF==1 and SFAmacrine==1:
                zero_index = [idx for idx, enum in enumerate([param in 'WFAmacrineSigmoidSFAmacrine' for param in conn_list]) if enum==True]
                if len(zero_index)>0:
                    if nonZeroStrength[zero_index[0]]:
                        uncomment = True

            if uncomment == True:
                out_lines = uncomment_block(line_num,out_lines)

            uncomment = False

            ## Make substitutions for desired param values
            indices = [idx for idx, enum in enumerate([param in line for param in conn_list]) if enum == True] #list of indices (locations in line) where word of interest (param) is located
            if len(indices) > 0: #if the current line has any of the parameters
                for lol_idx in indices:
                    if len(conn_lol[lol_idx])>1:
                        new_line = re.sub(conn_list[lol_idx],conn_lol[lol_idx][param_idx],out_lines[line_num],count=1)
                    else:
                        new_line = re.sub(conn_list[lol_idx],conn_lol[lol_idx][0],out_lines[line_num],count=1)
                    out_lines[line_num] = new_line
            if 'NUMSTEPS' in line:
                new_line = re.sub('NUMSTEPS',num_steps,line,count=0)
                out_lines[line_num] = new_line
            if 'OUTPATH' in line:
                new_line = re.sub('OUTPATH',full_results_path,line,count=0)
                out_lines[line_num] = new_line
            if 'PARAMSFILE' in line:
                new_line = re.sub('PARAMSFILE',idx_out_filename,line,count=0)
                out_lines[line_num] = new_line
            if 'INIMGPATH' in line:
                new_line = re.sub('INIMGPATH',input_image,line,count=0)
                out_lines[line_num] = new_line
            if 'INMOVPATH' in line:
                new_line = re.sub('INMOVPATH',input_movie,line,count=0)
                out_lines[line_num] = new_line
            if 'STOCHASTICRELFLAG' in line:
                new_line = re.sub('STOCHASTICRELFLAG',stochastic_flag,line,count=0)
                out_lines[line_num] = new_line
            if 'PREACTNOTRATE' in line:
                if stochastic_flag is '0':
                    new_line = re.sub('PREACTNOTRATE','1',line,count=0)
                elif stochastic_flag is '1':
                    new_line = re.sub('PREACTNOTRATE','0',line,count=0)
                else:
                    print("\ntune_params: STOCHASTICRELFLAG must be 0 or 1")
                    exit()
                out_lines[line_num] = new_line

        #####ENDFOR - line_num

        ##Write to output file
        print "tune_params: Writing new params file."
        try:
            out_fid = open(full_out_file,'w')
        except IOError as e:
            print "\ntune_params: Failed to open file "+full_out_file+" with error:\n"
            exit(e)
        for out_line in out_lines:
            out_fid.write("%s" % out_line)
        out_fid.close()

        ## Run petavision for this output file
        if run_PetaVision:
            print "tune_params: Running PetaVision.\n\n"
            os.system('mkdir -p '+full_results_path)
            mpi_cmd = '/opt/local/bin/openmpirun -np '+mpi_np
            if pipe_output_to_file:
                run_cmd = mpi_cmd+' '+run_path+' -rows '+mpi_rows+' -columns '+mpi_columns+' -p '+full_out_file+' > '+full_results_path+'/stdout.txt'
            else:
                run_cmd = mpi_cmd+' '+run_path+' -rows '+mpi_rows+' -columns '+mpi_columns+' -p '+full_out_file
            os.system('time '+run_cmd)
            os.system('cp '+full_out_file+' '+full_results_path)
            print "\n\ntune_params: Finished running PetaVision."

    #####ENDFOR - param_idx
#####ENDFOR - num_steps
#####ENDFUNCTION
