/*
 * pv_mex.cpp
 *
 *  Created on: Aug 16, 2008
 *      Author: dcoates
 */

/* PV <-> Octave/MATLAB Interface */

//#include "mex.h"

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "../pv_common.h"
#include "pv_ifc.h"

#include "../columns/HyPerCol.hpp"
#include "../layers/HyPerLayer.hpp"

#define MAX_PARAMS 32

// layer + connection handlers/callback headers
#include "../layers/LIF.h"
#include "../layers/LIF2.h"
#include "../layers/thru.h"
#include "../layers/gauss2D.h"
#include "../layers/gauss2Dx.h"
#include "../layers/cocirc1D.h"
#include "../layers/fileread.h"
#include "../layers/prob_fire.h"

extern "C" {

static int getHandler(int which, void **func, void **init ) {
	*func=NULL;
	*init=NULL;
	switch (which) {
	case PV_HANDLER_LIF:
		*func = (void*) &LIF_update;
		break;
	case PV_HANDLER_READFILE:
		*func = (void*) &fileread_update;
		*init = (void*) &fileread_init;
		break;
	case PV_HANDLER_GAUSS2D:
		*func = (void*) &gauss2D_rcv;
		*init = (void*) &gauss2D_init;
		break;
	case PV_HANDLER_GAUSS2DX:
		*func = (void*) &gauss2Dx_rcv;
		*init = (void*) &gauss2Dx_init;
		break;
#ifndef _MSC_VER
	case PV_HANDLER_COCIRC1D:
		*func = (void*) &cocirc1D_rcv;
		*init = (void*) &cocirc1D_init;
		break;
	case PV_HANDLER_PROB_FIRE:
		*func = (void*) &probFire_update;
#endif //MSVC
		break;
	case PV_HANDLER_LIF2:
// 		*func = (void*) &LIF2_update_explicit_euler;
		*func = (void*) &LIF2_update_exact_linear;
		*init = (void*) &LIF2_init;
		break;
	default:
		*func = NULL;
		*init = NULL;
		return -1;
	} // switch
	return 0;
}

int PV_ifc_getBufferPtr(void *hc, int layerOrConnection, int type, int which,
		float **buf, int *size) {
   if (layerOrConnection > PV_CONNECTION_FLAG) {
           int idx;
           idx = layerOrConnection - PV_CONNECTION_FLAG;
           //HyPerConnection *c = hc->connections(idx);
   } else {
       PVLayer* l = ((PV::HyPerCol*) hc)->getCLayer(layerOrConnection); // TODO err chking
           switch (type) {
           case PV_BUFFER_PHI:
              // TODO - take into account extended border
              *buf = l->phi[which];
              *size = l->numNeurons;
              break;
           case PV_BUFFER_V:
              *buf = l->V;
              *size = l->numNeurons;
              break;
           case PV_BUFFER_G_E:
              *buf = l->G_E;
              *size = l->numNeurons;
              break;
           case PV_BUFFER_G_I:
              *buf = l->G_I;
              *size = l->numNeurons;
              break;
           case PV_BUFFER_F:
              if (l->writeIdx>0)
                 *buf = l->fActivity[l->writeIdx - 1 % MAX_F_DELAY];
              else
                 *buf = l->fActivity[ MAX_F_DELAY - 1];
              *size = l->numNeurons;
              break;
           }
   }
   return 0;
}

// Useful helpers for all callers.
int PV_ifc_addLayer(void* hc, const char *name, int id, float dx, float dy,
		int features) {
	int layerNum;
// TODO - take care of layer id parameter n
	PVLayer *l = PVLayer_new(name, dx, dy, features);
	layerNum = ((PV::HyPerCol*) hc)->addCLayer(l);
	return layerNum;
}

// TODO - take care of layer id parameter n
int PV_ifc_setParams(void* hc, int n, int numParams, void* params,
		int which_func) {
	void *func, *init;
	PVLayer* l = ((PV::HyPerCol*) hc)->getCLayer(n); // TODO err chking
	if (getHandler(which_func, &func, &init)) {
		printf("ERROR: bad handler.\n");
		// TODO: what to do? how to bail well?
		return -1;
	}

	PVLayer_setParams(l, numParams, params);
        PVLayer_setFuncs(l, (UPDATE_FN) func, (INIT_FN)init);

	return 0;
}

int PV_ifc_connect(void* hc, int idxPre, int idxPost, int delay, int which_phi,
		int numParams, void* params, int which_func) {
	void *func, *init;
	if (getHandler(which_func, &func, &init)) {
		printf("ERROR: bad handler.\n");
		// TODO: what to do? how to bail well?
		return -1;
	}

	((PV::HyPerCol*) hc)->addCConnection(
			((PV::HyPerCol*) hc)->getCLayer(idxPre),
			((PV::HyPerCol*) hc)->getCLayer(idxPost), delay, which_phi,
			numParams, params, (UPDATE_FN) func, (INIT_FN) init);
	return 0;
}

void *PV_ifc_HyperCol_init( void )
{
	// TODO: pass in the global image specs
	PVRect imageRect = {0.0, 0.0, 64.0, 64.0};
	return new PV::HyPerCol (NULL, NULL, MAX_LAYERS, MAX_CONNECTIONS, imageRect);
}

int PV_ifc_HyperCol_finalize(void* hc)
{
	delete (PV::HyPerCol*)hc;	// call all it's destructors, etc.
	return 0;
}


int PV_ifc_run(void* hc, int steps)
{
	return ((PV::HyPerCol*)hc)->run(steps);
}


#if 0
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	static PV::HyPerCol* hc=NULL;

	/* Check for proper number of arguments */
	if (nrhs < 1)
	{
		mexPrintf("Need at least one arg. More doc TBD.\n");
		return;
	}

	double action = mxGetScalar(prhs[0]);

	if (action == 1.0)
	pyruntest(1);

	if (action == 5.0)
	pyruntest(5);

	return;
}

// Create a matrix for the return argument
// Need to populate result[0] later...
mxArray *result1 = mxCreateDoubleMatrix(1, 1, mxREAL);
double *output1 = mxGetPr(result1);
plhs[0] = result1;
mxArray *result2 = mxCreateDoubleMatrix(1, MAX_DIMS, mxREAL);
double *output2 = mxGetPr(result2);
plhs[1] = result2;
mxArray *result3 = mxCreateDoubleMatrix(1, MAX_DIMS, mxREAL);
double *output3 = mxGetPr(result3);
plhs[2] = result3;

double *pre = mxGetPr(prhs[0]);
double *post = mxGetPr(prhs[1]);
double *params = mxGetPr(prhs[2]);

int numPreDim = mxGetNumberOfElements(prhs[0]);
int numPostDim = mxGetNumberOfElements(prhs[1]);
int numParams = mxGetNumberOfElements(prhs[2]);

// Translate the params
float fParams[MAX_PARAMS];
int n;
for (n = 0; n < numParams; n++)
{
	fParams[n] = params[n];
}
float weight;

// Extract the position vectors
float prePos[MAX_DIMS], postPos[MAX_DIMS];
for (n=0; n<numPreDim; n++)
prePos[n] = pre[n];
for (n=0; n<numPostDim; n++)
postPos[n] = post[n];

//cocirc_calcWeight((cocirc_params*)&fParams, prePos, postPos, &weight);
//gabor_calcWeight((gabor_params*) &fParams, prePos, postPos, &weight);

usage();

// return the result
output1[0] = weight;

// Populate the positions
output2[0] = prePos[0];
output2[1] = prePos[1];
output2[2] = prePos[2];
output3[0] = postPos[0];
output3[1] = postPos[1];
output3[2] = postPos[2];
return;
}
#endif //0

} // extern "C"
