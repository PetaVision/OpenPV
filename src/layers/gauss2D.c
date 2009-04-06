/*
 * gauss2D.c
 *
 *  Created on: Aug 4, 2008
 *      Author: dcoates
 */

#include "gauss2D.h"
#include "../include/pv_common.h"
#include "../connections/WeightCache.h"

#include <stdlib.h>
#include <math.h>

#ifdef _MSC_VER
#define inline _inline
#endif

// Calculate the Gaussian "weight" between the two neurons.
// cells can be either oriented or unoriented
int gauss2D_calcWeight(PVLayer * pre, PVLayer * post, gauss2D_params *params,
      float* prePos, float* postPos, float *weight)
{

   // Get the Euclidean distance between the two points
   float dx = prePos[DIMX] - postPos[DIMX];
   float dy = prePos[DIMY] - postPos[DIMY];

   // Apply periodic boundary conditions
   if (GAUSS2D_PARAMS(G_usePBC)) {
      dx = fabs(dx) > NX / 2 ? -(dx / fabs(dx)) * (NX - fabs(dx)) : dx; // PBCs
      dy = fabs(dy) > NY / 2 ? -(dy / fabs(dy)) * (NY - fabs(dy)) : dy;
   }

   float theta;
   *weight = 1.0;
   if ((pre->numFeatures >= NO) && (post->numFeatures == 1)) {
      int preNk = pre->numFeatures / NO;
      theta = ((int) prePos[DIMO] / preNk) * DTH * DEG_TO_RAD;
   }
   else if ((pre->numFeatures == 1) && (post->numFeatures >= NO)) {
      int postNk = post->numFeatures / NO;
      theta = ((int) postPos[DIMO] / postNk) * DTH * DEG_TO_RAD;
   }
   else if ((pre->numFeatures >= NO) && (post->numFeatures >= NO)) {
      int preNk = pre->numFeatures / NO;
      int postNk = post->numFeatures / NO;
      theta = (((int) postPos[DIMO] / postNk) - ((int) prePos[DIMO] / preNk)) * DTH
            * DEG_TO_RAD;
      *weight = exp(-theta * theta / (DTH * DTH));
      theta = 0.0;
   }
   else theta = 0.0;

   float dxp = dx * cos(theta) + dy * sin(theta);
   float dyp = -1.0 * dx * sin(theta) + dy * cos(theta);
   float aspect = GAUSS2D_PARAMS(G_ASPECT);
   float d2 = dxp * dxp + (aspect * aspect) * dyp * dyp;

   if (d2 > GAUSS2D_PARAMS(G_R2)) {
      *weight = 0;
      return 0;
   }
   float sigma = GAUSS2D_PARAMS(G_SIGMA);
   *weight *= exp(-d2 / (sigma * sigma));

   return 1;
}//gauss2D_calcWeight

// uses the graded input activity as a probability for generating post synaptic events
// functionally equivalent to gauss2D_rcv when input activity is binary {0,1} but much slower
int gauss2D_graded_rcv(PVConnection* con, PVLayer *post, int nActivity, float *fActivity)
{
   PVLayer* pre = con->pre;
   // TODO - take into account extended border
   float *phi = con->post->phi[con->whichPhi];

   int postNdx, preNdx;
   float prePos[MAX_DIMS], postPos[MAX_DIMS];
   float weight;
   int numSynapses, postCounter;
   int postKernelIndex, preKernelIndex;
   int postIdxOffset = (post->yOrigin * post->loc.nx + post->xOrigin) * post->numFeatures;

   // For each neuron in the presynaptic patch
   for (preNdx = 0; preNdx < pre->numNeurons; preNdx++) {
      if (fActivity[preNdx] == 0.0) continue; // optimization: skip 0 inputs (spiking or continuous)

      pvlayer_getPos(pre, preNdx, &prePos[DIMX], &prePos[DIMY], &prePos[DIMO]);
      preKernelIndex = PV_weightCache_getPreKernelIndex(con, prePos, 1);
      PV_weightCache_getPostNeurons(con, prePos, preKernelIndex, &numSynapses);

      for (postCounter = 0; postCounter < numSynapses; postCounter++) {
         if (rand() > RAND_MAX * fActivity[preNdx]) continue; // test each synapse for quantal release

         PV_weightCache_getPostByIndex(con, prePos, preKernelIndex, postCounter, &weight,
               postPos, &postKernelIndex);
         if (pvlayer_getIndex(post, postPos, &postNdx) < 0) continue;
         // TODO - take into account extended border
         if (weight) phi[postNdx + postIdxOffset] += weight
               * con->preNormF[preKernelIndex] * con->postNormF[postKernelIndex];
      } // for each input neuron
   }
   return 0;
}

// Call default handler to loops over presynaptic spikes using weightCache and
// normalization.
int gauss2D_rcv(PVConnection* con, PVLayer *post, int nActivity, float *fActivity)
{
   return PVConnection_default_rcv(con, post, nActivity, fActivity);
}

// Calc responses to uniform stimuli to normalize the total input into each postsynaptic neuron
// to account for pixel aliasing.
int gauss2D_calc_normalize(PVConnection* con)
{
   const void *params = con->params;
   return PVConnection_default_normalize(con, (PVWeightFunction) gauss2D_calcWeight,
         GAUSS2D_PARAMS(GAUSS2D_WEIGHT_SCALE));
}//gauss2D_calc_normalize


int gauss2D_init(PVConnection* con)
{
   const void *params = con->params;
   con->r2 = GAUSS2D_PARAMS(G_R2);
   gauss2D_calc_normalize(con);
   return 0;
}
