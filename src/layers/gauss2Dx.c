/*
 * gauss2Dx.c
 *
 *  Created on: Aug 4, 2008
 *      Author: dcoates
 */

#include "gauss2Dx.h"

#include "../include/pv_common.h"
#include "../connections/WeightCache.h"

#include <stdlib.h>
#include <math.h>

#ifdef _MSC_VER
#define inline _inline
#endif

// Calculate the "weight" between the two neurons.
int gauss2Dx_calcWeight(PVLayer * pre, PVLayer * post, gauss2Dx_params *params,
      float* prePos, float* postPos, float *weight)
{
   float theta, deltaTheta;
   *weight = 0.0;
   if (pre->numFeatures >= NO && post->numFeatures >= NO) {
      int postNk = post->numFeatures / NO;
      float postTheta = ((int) postPos[DIMO] / postNk) * DTH; //deg
      int preNk = pre->numFeatures / NO;
      float preTheta = ((int) prePos[DIMO] / preNk) * DTH; //deg
      deltaTheta = fabs(preTheta - postTheta); //deg
      theta = preTheta;
   }
   else if (pre->numFeatures >= NO && post->numFeatures == 1) {
      int preNk = pre->numFeatures / NO;
      theta = ((int) prePos[DIMO] / preNk) * DTH; //deg
      deltaTheta = theta;
   }
   else if (pre->numFeatures == 1 && post->numFeatures >= NO) {
      int postNk = post->numFeatures / NO;
      theta = ((int) postPos[DIMO] / postNk) * DTH; //deg
      deltaTheta = theta;
   }
   else {
      theta = 0.0;
      deltaTheta = theta;
   }
   deltaTheta = deltaTheta <= 90. ? deltaTheta : 180. - deltaTheta;
   if (deltaTheta > GAUSS2DX_PARAMS(G_DTH_MAX)) {
      return 0;
   }
   float sigmaTheta2 = GAUSS2DX_PARAMS(G_SIGMA_THETA2);
   *weight = exp(-deltaTheta * deltaTheta / sigmaTheta2);

   // Get the Euclidean distance between the two points
   float dx = prePos[DIMX] - postPos[DIMX];
   float dy = prePos[DIMY] - postPos[DIMY];

   // Apply periodic boundary conditions
   if (GAUSS2DX_PARAMS(G_usePBC)) {
      dx = fabs(dx) > NX / 2 ? -(dx / fabs(dx)) * (NX - fabs(dx)) : dx; // PBCs
      dy = fabs(dy) > NY / 2 ? -(dy / fabs(dy)) * (NY - fabs(dy)) : dy;
   }

   theta *= DEG_TO_RAD;
   float dxp = dx * cos(theta) + dy * sin(theta);
   float dyp = -1.0 * dx * sin(theta) + dy * cos(theta);
   if (GAUSS2DX_PARAMS(G_ASYM_FLAG) > 0) dyp = dyp - GAUSS2DX_PARAMS(G_OFFSET);
   if (GAUSS2DX_PARAMS(G_ASYM_FLAG) < 0) dyp = dyp + GAUSS2DX_PARAMS(G_OFFSET);

   float aspect = GAUSS2DX_PARAMS(G_ASPECT);
   float d2 = dxp * dxp + aspect * aspect * dyp * dyp;
   if (d2 > GAUSS2DX_PARAMS(G_R2)) {
      *weight = 0;
      return 0;
   }

   float sigma = GAUSS2DX_PARAMS(G_SIGMA);
   *weight *= exp(-d2 / (sigma * sigma));

   return 1;
} //  gauss2Dx_calcWeight

// Call default handler to loop over presynaptic spikes using weightCache and
// normalization.
int gauss2Dx_rcv(PVConnection* con, PVLayer *post, int nActivity, float *fActivity)
{
   return PVConnection_default_rcv(con, post, nActivity, fActivity);
}

// Calc responses to uniform stimuli to normalize the total input into each postsynaptic neuron
// to account for pixel aliasing.
int gauss2Dx_calc_normalize(PVConnection* con)
{
   const void *params = con->params;
   return PVConnection_default_normalize(con, (PVWeightFunction) gauss2Dx_calcWeight,
         GAUSS2DX_PARAMS(G_WEIGHT_SCALE));
}//gauss2D_calc_normalize


int gauss2Dx_init(PVConnection* con)
{
   const void *params = con->params;
   con->r2 = GAUSS2DX_PARAMS(G_R2); // TODO: can we figure out a good worst-case to filter?
   gauss2Dx_calc_normalize(con);
   return 0;
}
