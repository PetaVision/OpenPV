/*
 * pv_mex.c
 *
 *  Created on: Aug 16, 2008
 *      Author: dcoates
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <mex.h>

#include "../pv_common.h"
#include "pv_ifc.h"

int mxToFloats(mxArray *mxa, float **a)
{
	int num = mxGetNumberOfElements(mxa);
	double *arr = mxGetPr(mxa);
	int i;

	*a = (float*) mxMalloc(num * sizeof(float)); //this can be freed, which MATLAB should do

	for (i = 0; i < num; i++)
		(*a)[i] = arr[i];

	return 0;
}

int mxToChars(mxArray *mxa, char *a)
{
	int num = mxGetN(mxa);
	int i;

	mxGetString(mxa,a,num);
	return 0;
}


int mxInject(mxArray *mxa, float *a)
{
	int num = mxGetNumberOfElements(mxa);
	double *arr = mxGetPr(mxa);
	int i;

	for (i = 0; i < num; i++)
		a[i] = arr[i];

	return 0;
}

int mxMeasure(mxArray *mxa, float *a, int size)
{
	int i;

	double *cptr = mxGetPr(mxa);
	printf("size = %d, ptr=%x\n", size, cptr);
	for (i = 0; i < size; i++)
		cptr[i] = a[i];

	return 0;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	static void *hc=NULL;
	int action = (int) mxGetScalar(prhs[0]);
	int ret, steps, num;
	double *arg1;
	float *params;
	float *buf;
	int size;
	mxArray *resulta;
	char filename[128];
	mxArray *result1;
	double *output1;

	strcpy(filename,"io/input/t64_input.bin");


	if (nrhs < 1)
	{
		mexPrintf("PetaVision MATLAB/Octave MEX Interface.\n");
		return;
	}

	nlhs=1;

	switch (action)
	{
	case PV_ACTION_INIT:
		if (hc)
			PV_ifc_HyPerCol_finalize(hc);
		hc = PV_ifc_HyPerCol_init();
		ret = 1;
		// TODO: output the params.txt file
		system("rm -fr output/*");
		break;

	case PV_ACTION_FINALIZE:
		ret = PV_ifc_HyPerCol_finalize(hc);
		break;

	case PV_ACTION_ADD_LAYER:
		// TODO: Pass strings
		arg1 = mxGetPr(prhs[1]);
		ret = PV_ifc_addLayer(hc, "Layer", arg1[0], arg1[1], arg1[2], arg1[3]);
		break;

	case PV_ACTION_SET_LAYER_PARAMS:
		arg1 = mxGetPr(prhs[1]);
		num = mxGetNumberOfElements(prhs[2]);
		mxToFloats((mxArray*) prhs[2], &params);
		// Hack: need a better way to get in strings. TODO
		if (arg1[1] == PV_HANDLER_READFILE)
		{
			(((void**)params)[6]) = filename;
		}
		PV_ifc_setParams(hc, arg1[0], num, (void*) params, arg1[1]);
		ret = 1;
		break;

	case PV_ACTION_ADD_CONNECTION:
		arg1 = mxGetPr(prhs[1]);
		num = mxGetNumberOfElements(prhs[2]);
		mxToFloats((mxArray*) prhs[2], &params);
		PV_ifc_connect(hc, arg1[0], arg1[1], arg1[2], arg1[3],
 			num, (void*) params, arg1[4]);
		ret = 1;
		break;

	case PV_ACTION_RUN:
		steps = (int) mxGetScalar(prhs[1]);
		PV_ifc_run(hc, steps);
		ret = 1;
		break;

	case PV_ACTION_SET_PARAMS:
		break;

	case PV_ACTION_SET_INPUT_FILENAME:
		mxToChars((mxArray*) prhs[1], filename);
		break;

	case PV_ACTION_INJECT:
		arg1 = mxGetPr(prhs[1]);
		// TODO: could directly inject into destination buffer, w/o two steps
		mxToFloats((mxArray*) prhs[2], &params);
		PV_ifc_getBufferPtr(hc, arg1[0],arg1[1],arg1[2], &buf, &size);
		mxInject((mxArray*) prhs[2], buf);
		break;

	case PV_ACTION_MEASURE:
		arg1 = mxGetPr(prhs[1]);
		PV_ifc_getBufferPtr(hc, arg1[0],arg1[1],arg1[2], &buf, &size);
		resulta = mxCreateDoubleMatrix(1, size, mxREAL);
		mxMeasure(resulta, buf, size);
		plhs[1] = resulta;
		nlhs=2;
		break;

	default:
		break;
	} // switch

	result1 = mxCreateDoubleMatrix(1, 1, mxREAL);
	output1 = mxGetPr(result1);
	output1[0] = ret;
	plhs[0] = result1;

	return;

}
