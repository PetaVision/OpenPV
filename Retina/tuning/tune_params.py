#!/usr/bin/env python

import os
import re
from tune_functions import *

## Global vars
param_file  = 'params_text.pv'
output_file = 'generated_params'
mpi_np      = '1'
wrkspc_path = '~/Documents/Work/LANL/workspace/'

input_path  = wrkspc_path+'Retina/input/'
run_path    = wrkspc_path+'Retina/Debug/Retina'

## Declare params - frange is (start, end, int)
ImageRetina                 = ["%g" % x for x in frange(0,1,1)]
ImageCone                   = ["%g" % x for x in frange(0,1,1)]
ConeSigmoidBipolar          = ["%g" % x for x in frange(0,1,1)]
ConeSigmoidHorizontal       = ["%g" % x for x in frange(0,1,1)]
HoriGapHorizontal           = ["%g" % x for x in frange(0,1,1)]
HoriSigmoidCone             = ["%g" % x for x in frange(0,1,1)]
BipolarSigmoidGanglion      = ["%g" % x for x in frange(0,1,1)]
BipolarSigmoidWFAmacrine    = ["%g" % x for x in frange(0,1,1)]
WFAmacrineGapWFAmacrine     = ["%g" % x for x in frange(0,1,1)]
WFAmacrineSigmoidBipolar    = ["%g" % x for x in frange(0,1,1)]
GanglionGapPAAmacrine       = ["%g" % x for x in frange(0,1,1)]
PAAmaGapGanglion            = ["%g" % x for x in frange(0,1,1)]
PAAmaGapPAAmacrine          = ["%g" % x for x in frange(0,1,1)]
PAAmacrineGanglion          = ["%g" % x for x in frange(0,1,1)]
PAAmacrinePAAmacrine        = ["%g" % x for x in frange(0,1,1)]
PAAmacrineWFAmacrine        = ["%g" % x for x in frange(0,1,1)]
BipolarSigmoidSFAmacrine    = ["%g" % x for x in frange(0,1,1)]
SFAmacrineSigmoidGanglion   = ["%g" % x for x in frange(0,1,1)]
SFAmacrineSigmoidPAAmacrine = ["%g" % x for x in frange(0,1,1)]
WFAmacrineSFAmacrine        = ["%g" % x for x in frange(0,1,1)]
GanglionSynchronicity       = ["%g" % x for x in frange(0,1,1)]

param_list = ["ImageRetina","ImageCone","ConeSigmoidBipolar","ConeSigmoidHorizontal",
        "HoriGapHorizontal","HoriSigmoidCone","BipolarSigmoidGanglion","BipolarSigmoidWFAmacrine",
        "WFAmacrineGapWFAmacrine","WFAmacrineSigmoidBipolar","GanglionGapPAAmacrine",
        "PAAmaGapGanglion","PAAmaGapPAAmacrine","PAAmacrineGanglion","PAAmacrinePAAmacrine",
        "PAAmacrineWFAmacrine","BipolarSigmoidSFAmacrine","SFAmacrineSigmoidGanglion",
        "SFAmacrineSigmoidPAAmacrine","WFAmacrineSFAmacrine","GanglionSynchronicity"]

param_lol = [ImageRetina,ImageCone,ConeSigmoidBipolar,ConeSigmoidHorizontal,
        HoriGapHorizontal,HoriSigmoidCone,BipolarSigmoidGanglion,BipolarSigmoidWFAmacrine,
        WFAmacrineGapWFAmacrine,WFAmacrineSigmoidBipolar,GanglionGapPAAmacrine,
        PAAmaGapGanglion,PAAmaGapPAAmacrine,PAAmacrineGanglion,PAAmacrinePAAmacrine,
        PAAmacrineWFAmacrine,BipolarSigmoidSFAmacrine,SFAmacrineSigmoidGanglion,
        SFAmacrineSigmoidPAAmacrine,WFAmacrineSFAmacrine,GanglionSynchronicity]

## Assert that all parameter lists are the same length
if not all(len(i) == len(param_lol[0]) for i in param_lol):
    exit("\n\ntune_params: ERROR: One of the lists is not the right size!\n")

## Open file
if os.path.isfile(param_file):
    try:
        print "tune_params: Opening param file "+param_file+"."
        in_fid = open(param_file)
        param_lines = in_fid.readlines()
        in_fid.close()
    except IOError as e:
        print "tune_params: Failed to open file "+param_file+" with error:\n"+e
else:
    exit("\n\ntune_params: ERROR: Couldn't find file "+param_file+"!\n")

## Mega for-loop
for param_idx in range(len(param_lol[0])):
    print "tune_params: Writing new params ("+str(param_idx)+" of "+str(len(param_lol[0])-1)+")."
    out_filename = output_file+str(param_idx)+'.pv'
    try:
        out_fid = open(out_filename,'w')
    except IOError as e:
        print "tune_params: Failed to open file "+output_file+" with error:\n"+e
    for line in param_lines:
        indices = [idx for idx, enum in enumerate([param in line for param in param_list]) if enum == True]
        if len(indices) > 0: #if the current line has any of the parameters
            for lol_idx in indices:
                new_line = re.sub(param_list[lol_idx],param_lol[lol_idx][param_idx],line,count=1)
                out_fid.write(new_line)
        else:
            out_fid.write(line)
    out_fid.close()
    os.system('cp '+out_filename+' '+input_path+out_filename)
    ## Run petavision for this output file
    print "tune_params: Running PetaVision"
    run_cmd = '/opt/local/bin/openmpirun -np '+mpi_np+' '+run_path+' -p '+input_path+out_filename
    os.system(run_cmd)
