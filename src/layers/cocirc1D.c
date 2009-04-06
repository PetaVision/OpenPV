/*
 * cocirc1D.c
 *
 *  Created on: Aug 4, 2008
 *      Author: dcoates
 */

#include "cocirc1D.h"

#include "../include/pv_common.h"
#include "../connections/WeightCache.h"

#include <stdlib.h>
#include <math.h>

#ifdef _MSC_VER
#define inline _inline
#endif

// Calculate the "weight" between the two neurons.
int cocirc1D_calcWeight(PVLayer *pre, PVLayer *post, cocirc1D_params *params,
      float* prePos, float* postPos, float *weight)
{
   *weight = 0;

   int postNk = post->numFeatures / NO;
   int preNk = pre->numFeatures / NO;

   // Get the Euclidean distance between the two points
   float dx = prePos[DIMX] - postPos[DIMX];
   float dy = prePos[DIMY] - postPos[DIMY];

   // Apply periodic boundary conditions
   if (COCIRC_PARAMS(COCIRC_usePBC)) {
      dx = fabs(dx) > NX / 2 ? -(dx / fabs(dx)) * (NX - fabs(dx)) : dx; // PBCs
      dy = fabs(dy) > NY / 2 ? -(dy / fabs(dy)) * (NY - fabs(dy)) : dy;
   }

   float d2 = dx * dx + dy * dy;
   if (d2 > COCIRC_PARAMS(COCIRC_R2)) return 0;
   float gDist = expf(-d2 / COCIRC_PARAMS(COCIRC_SIGMA_DIST2));

   // TODO: precompute sin/cos
   float preTheta = preNk > 0 ? ((int) prePos[DIMO] / preNk) * DTH * DEG_TO_RAD : 0.0; //rad
   float postTheta = postNk > 0 ? ((int) postPos[DIMO] / postNk) * DTH * DEG_TO_RAD
         : preTheta;
   float gCocirc = 1.0;
   float gKurvePre = 1.0;
   float gKurvePost = 1.0;

   if (d2 == 0) {
      if (!COCIRC_PARAMS(COCIRC_SELF)) return 0;
      float deltaTheta = RAD_TO_DEG * fabs(preTheta - postTheta);
      deltaTheta = deltaTheta <= 90. ? deltaTheta : 180. - deltaTheta;
      gCocirc = expf(-deltaTheta * deltaTheta / COCIRC_PARAMS(COCIRC_SIGMA_COCIRC2));
   }
   else {
      float dxP = (dx * cos(preTheta) + dy * sin(preTheta));
      float dyP = (dy * cos(preTheta) - dx * sin(preTheta));
      float atanx2 = preTheta + 2. * atan2f(dyP, dxP); //preferred angle (rad)
      atanx2 += 2 * PI;
      atanx2 = fmod(atanx2, PI );
      float chi = RAD_TO_DEG * fabs(atanx2 - postTheta); //deg
      if (chi >= 90.) chi = 180. - chi;
      gCocirc = postNk > 0 && preNk ? expf(-chi * chi / COCIRC_PARAMS(COCIRC_SIGMA_COCIRC2)) : 1.0;
      float cocircKurve = fabs(2 * dyP) / d2;
      float preKurve = preNk > 0 ? DK * ((int) prePos[DIMO] % preNk) : 0.0;
      gKurvePre = preNk > 1 ? exp(-pow((cocircKurve - fabs(preKurve)), 2)
            / COCIRC_PARAMS(COCIRC_SIGMA_KURVE2)) : 1.0;
      float postKurve = postNk > 0 ? DK * ((int) postPos[DIMO] % postNk) : 0.0;
      gKurvePost = postNk > 1 && preNk > 1 ? exp(-pow((cocircKurve - fabs(postKurve)), 2)
            / COCIRC_PARAMS(COCIRC_SIGMA_KURVE2)) : 1.0;
   }

   *weight = gDist * gKurvePre * gKurvePost * gCocirc;
   return 1;
}

// Call default handler to loops over presynaptic spikes using weightCache and
// normalization.
int cocirc1D_rcv(PVConnection* con, PVLayer *post, int nActivity, float *fActivity)
{
   return PVConnection_default_rcv(con, post, nActivity, fActivity);
}

// Calc responses to uniform stimuli to normalize the total input into each postsynaptic neuron
// to account for pixel aliasing.
int cocirc1D_calc_normalize(PVConnection* con)
{
   const void *params = con->params;
   return PVConnection_default_normalize(con, (PVWeightFunction) cocirc1D_calcWeight,
         COCIRC_PARAMS(COCIRC_WEIGHT_SCALE));
}

int cocirc1D_init(PVConnection* con)
{
   const void *params = con->params;
   con->r2 = COCIRC_PARAMS(COCIRC_R2); // TODO: can we figure out a good worst-case to filter?
   cocirc1D_calc_normalize(con);
   return 0;
}
