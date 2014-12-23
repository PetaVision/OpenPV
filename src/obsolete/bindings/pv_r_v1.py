#!/usr/bin/python
#
# pv.py
#
# Created on: Aug 17, 2008
#     Author: dcoates
#

from numpy import *

# Python script to run Petavision

NX = 64
NY = 64
NO = 8
NK = 4

NOISE_AMP = 0.0
V_REST = -70.0
V_EXC = 0.0
V_INH = -75.
V_INHB = -90.
VTH_REST = (V_REST + 5.0) + 100.0
TAU_VMEM = 50.0
TAU_EXC = 2.0
TAU_INH = 5.0
TAU_INHB = 10.0
TAU_VTH = 20.0
DELTA_VTH = 5.0
GLOBAL_GAIN = 1.0
ASPECT_RATIO = 3.0
SIGMA_EDGE = 1.5
R2_COCIRC = 16.0 # 8.0
SIGMA_DIST_COCIRC = 8.0 # 4.0
R2_FEEDBACK = 6.0 # 8.0
SIGMA_DIST_FEEDBACK = 3.0 # 4.0

from pv_ifc import *

LAYER_R = 0
LAYER_V1 = 1

# Execution:
import pypv
#pypv.command(PV_ACTION_INIT, 4, 'pv -i io/input/amoeba2X.bin -w -n2')   
pypv.command(PV_ACTION_INIT, 3, 'pv -w -n2')   

pypv.command(PV_ACTION_ADD_LAYER, (LAYER_R, 1.0, 1.0, 1.0), 0)
pypv.command(PV_ACTION_ADD_LAYER, (LAYER_V1, 1.0, 1.0, NO*NK), 0)

# init
pypv.command(PV_ACTION_SET_LAYER_PARAMS, (LAYER_R, PV_HANDLER_READFILE),
             (0, 1.0, 1.0, 1.0, 1*0.999, 0*0.01, 20.0, 40.0, 0))

pypv.command(PV_ACTION_SET_LAYER_PARAMS, (LAYER_V1, PV_HANDLER_LIF2),
				   (V_REST, V_EXC, V_INH, V_INHB, 		
				   TAU_VMEM, TAU_EXC, TAU_INH, TAU_INHB, 
				   VTH_REST,  TAU_VTH, DELTA_VTH + 10.0,
				   0.25, NOISE_AMP*( 1.0/TAU_EXC ) * ( ( TAU_INH * (V_REST-V_INH) + TAU_INHB * (V_REST-V_INHB) ) / (V_EXC-V_REST) ), 
				   0.25, NOISE_AMP*1.0, 
				   0.25, NOISE_AMP*1.0))

pypv.command(PV_ACTION_ADD_CONNECTION, (LAYER_R, LAYER_V1, 0, 0, PV_HANDLER_GAUSS2D),
				   (1.0,
					2*2+2*2,
					SIGMA_EDGE,
					ASPECT_RATIO,
					5.0*GLOBAL_GAIN))

pypv.command(PV_ACTION_SETUP, 0, 0)   

inSpike = [ 0 for x in range(0,NY*NX)]
inSpike[(NY/2 - 2) * NX + (NX/2 + 0)] = 1.0;
inSpike[(NY/2 - 1) * NX + (NX/2 + 0)] = 1.0;
inSpike[(NY/2 + 0) * NX + (NX/2 + 0)] = 1.0;
inSpike[(NY/2 + 1) * NX + (NX/2 + 0)] = 1.0;
inSpike[(NY/2 + 2) * NX + (NX/2 + 0)] = 1.0;
pypv.command(PV_ACTION_INJECT, (LAYER_R, PV_BUFFER_V, 0), inSpike)   

pypv.command(PV_ACTION_RUN, 2, 0)   
pypv.command(PV_ACTION_FINALIZE, 0, 0)   
