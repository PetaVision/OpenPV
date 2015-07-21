/*
 * mexWeightHarness.c
 *
 *  Created on: Aug 5, 2008
 *      Author: dcoates
 */

// This stub allows usage of a layer's "calcWeight" function from
// inside Octave/MATLAB, particularly useful when tuning receptive
// fields.

#include "mex.h"

#include <stdlib.h>
#include <math.h>

#include "../src/pv_common.h"
#include "../src/layers/HyPerLayer.h"

#include "../src/layers/gabor.h"
#include "../src/layers/cocirc.h"

// For now, include the C file containing the code.
// TODO: Need to figure out how to link against Petavision.
//#include "../src/layers/gabor.c"
#include "../src/layers/cocirc.c"

#define MAX_DIMS 10
#define MAX_PARAMS 32

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// Check for proper number of arguments
	if (nrhs != 3)
	{
		mexPrintf("Three input arguments required: preIndex, postIndex, and layer params.\n");
		mexPrintf("Three output arguments given: weight, prePos, and postPos\n");
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

	cocirc_calcWeight((cocirc_params*)&fParams, prePos, postPos, &weight);
	//gabor_calcWeight((gabor_params*) &fParams, prePos, postPos, &weight);

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
