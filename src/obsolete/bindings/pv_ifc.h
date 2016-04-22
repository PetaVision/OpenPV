/*
 * pv_ifc.h
 *
 *  Created on: Aug 16, 2008
 *      Author: dcoates
 */

/* High-level, single-entry point interface to Petavision routines from outside */
#ifndef PV_IFC_H_
#define PV_IFC_H_

/* These need to be kept in sync with appropriate Octave/MATLAB/Python versions: */
#define PV_ACTION_INIT 1
#define PV_ACTION_ADD_LAYER 2
#define PV_ACTION_SET_LAYER_PARAMS 3
#define PV_ACTION_ADD_CONNECTION 4
#define PV_ACTION_RUN 5
#define PV_ACTION_SET_PARAMS 6
#define PV_ACTION_SET_INPUT_FILENAME 7
#define PV_ACTION_INJECT 8
#define PV_ACTION_MEASURE 9
#define PV_ACTION_FINALIZE 10
#define PV_ACTION_SETUP 11

#define PV_HANDLER_LIF 1
#define PV_HANDLER_READFILE 2
#define PV_HANDLER_GAUSS2D 3
#define PV_HANDLER_THRU 4
#define PV_HANDLER_COCIRC1D 5
#define PV_HANDLER_COCIRC_K 6
#define PV_HANDLER_CENTER_SURR 7
#define PV_HANDLER_PROB_FIRE 8
#define PV_HANDLER_LIF2 9
#define PV_HANDLER_GAUSS2DX 10
#define PV_HANDLER_COCIRC_K2 11

#define PV_BUFFER_V 0
#define PV_BUFFER_PHI 1
#define PV_BUFFER_G_I 2
#define PV_BUFFER_G_E 3
#define PV_BUFFER_F 4

#define PV_CONNECTION_FLAG 100

#ifdef __cplusplus
extern "C"
{
#endif

/* Helper functions for use by all interfaces. */
void* PV_ifc_HyPerCol_init( void );
int PV_ifc_HyPerCol_finalize( void* hc );

int PV_ifc_addLayer(void* hc, const char *name, int id, float dx, float dy, int features);
int PV_ifc_setParams(void* hc, int n, int numParams, void* params,
		int which_func);
int PV_ifc_connect(void* hc, int idxPre, int idxPost, int delay, int which_phi,
		int numParams, void* params, int which_func);
int PV_ifc_getBufferPtr(void *hc, int layerOrConnection, int type, int which, float **buf, int* size);
int PV_ifc_run(void* hc, int steps);

#ifdef __cplusplus
}
#endif

#endif /* PV_IFC_H_ */
