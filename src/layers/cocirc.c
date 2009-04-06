// ---------------------------------------------------
// Co-circularity connection routines
// ---------------------------------------------------

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "cocirc.h"
#include "../include/pv_common.h"

#define PARAMS(p) (params->p)

// Helper function to get 3-dimensional (x,y,theta) coordinates from
// a index.
static inline void getPos3(int idx, float* coords)
{
   coords[DIMY] = (idx / (NX * NO));
   coords[DIMX] = ((idx / NO) % NX);
   coords[DIMO] = (idx % NO);
}

// Helper function to get 2-dimensional (x,y) coordinates from idx
static inline void getPos2(int idx, float* coords)
{
   coords[DIMY] = (idx / NX);
   coords[DIMX] = (idx % NX);
}

static inline int cocirc_calcWeight(cocirc_params* params, float *prePos, float* postPos,
      float*weight)
{
   float dx, dy, d2, gd, gt, ww;
   float chi;

   // Calculate the distance between the two neurons' x and y locations
   dx = prePos[DIMX] - postPos[DIMX];
   dy = prePos[DIMY] - postPos[DIMY];

   // If desired, apply periodic boundary conditions
   if (PARAMS(CC_usePBC)) {
      dx = fabs(dx) > PARAMS(CC_NX) / 2 ? -(dx / fabs(dx)) * (PARAMS(CC_NX) - fabs(dx)) : dx; // PBCs
      dy = fabs(dy) > PARAMS(CC_NY) / 2 ? -(dy / fabs(dy)) * (PARAMS(CC_NY) - fabs(dy)) : dy;
   }

   d2 = dx * dx + dy * dy;

   if (d2 > PARAMS(CC_R2)) {
      // For now, inhibit anyone outside our range (for winner-take-all)
      *weight = PARAMS(CC_DISTANT_VAL);
      return 0;
   }

   // Translate delta x and delta y based on post's orientation
   float theta = prePos[DIMO] * PARAMS(CC_DTH) * PI / 180.0;
   float xp = dx * cos(theta) + dy * sin(theta);
   float yp = -1.0 * dx * sin(theta) + dy * cos(theta);

#if 1
   float alpha_p = RAD_TO_DEG_x2 * atan2f(yp, xp) / 1.0;
   float theta_p = prePos[DIMO] * PARAMS(CC_DTH) + alpha_p;
   chi = theta_p - postPos[DIMO] * PARAMS(CC_DTH);

   float rawchi = chi; // for debugging
   chi = abs(chi);
   chi = fmodf(chi, 360.0f);
   if (chi >= 180.0f) chi = 360.0f - chi;
#else // Gar's suggestion:
   float alpha_p = RAD_TO_DEG_x2 * atan2f(yp, xp);
   float theta_p = prePos[DIMO] * PARAMS(CC_DTH) - alpha_p;

   float rawchi = chi; // for debugging
   chi = postPos[DIMO] * PARAMS(CC_DTH) + alpha_p;
   chi += 360.0;
   chi = fmodf(chi, 180.0f);
   if (chi >= 90.0f)
   chi = 180.0f - chi;
#endif
#if 0
   // TODO: Get this working!
   // Apply maximum curvature constraint using the length of the chord.
   //float chord_length = (theta_p_correct == 0) ? 0 : sqrt(d2) / abs(theta_p_correct / 180.0);
   float chord_length, curv, radius;
   radius = sqrt(d2)/ (2.0 * sin(alpha_p));
   curv = exp(-1.0 / abs(radius) / PARAMS(CC_SIG_CURV));
#else
   float curv = 1.0, radius = 0.0;
#endif

   // TODO: Make a macro so that in release builds we don't even check the print param: just noop
   if (PARAMS(CC_DEBUG)) printf("yxo=%3.3f %3.3f %3.3f yxo=%3.3f %3.3f %3.3f "
      "th=%f, theta'=%f, radius=%f curv=%f chi=%f yp=%f xp=%f, alpha_p=%f raw_chi=%f\n",
         prePos[DIMY], prePos[DIMX], prePos[DIMO], postPos[DIMY], postPos[DIMX],
         postPos[DIMO], theta, theta_p, radius, curv, chi, yp, xp, alpha_p, rawchi);

   // Apply Gaussians
   gt = expf(-chi * chi / PARAMS(CC_SIG_C_P_x2));
   gd = expf(-d2 / PARAMS(CC_SIG_C_D_x2));

   // Calculate and apply connection efficacy/weight
   ww = PARAMS(CC_COCIRC_SCALE) * gd * curv * (gt - PARAMS(CC_INHIB_FRACTION));
   ww = (ww < 0.0) ? ww * PARAMS(CC_INHIBIT_SCALE) : ww;

   *weight = ww;
   return 1;
}
;

// --------------------------------------------------------------------------
// Default rcvSynapticInput() implementation:
//
// Input: non-sparse activity input.
// Output: For each postsynaptic, neuron, sum of weights based on input activity.
//
// Algorithm: Finds each active presynaptic neurons and calculates weight for
// each post-synaptic neuron, summing weights to get a single value for each
// post-synaptic neuron.
// --------------------------------------------------------------------------
int cocirc_rcv(PVLayer* pre, PVLayer* post, float *phi, int nActivity, float *fActivity,
      void *params)
{
   int i, j;
   float prePos[MAX_DIMS], postPos[MAX_DIMS];
   float weight;

   // For each neuron in the presynaptic patch
   for (j = 0; j < pre->numNeurons; j++) {
      if (fActivity[j] == 0.0) continue; // optimization: skip 0 inputs (spiking or continuous)

      // Determine presynaptic neuron's features
      // TODO: Need to translate the pre vs. post column
      getPos3(j, prePos);

      // For each neuron in the postsynaptic patch
      for (i = 0; i < post->numNeurons; i++) {
         // Determine postsynaptic feature vector
         getPos3(i, postPos);

         // Call the weight calculation handler:
         cocirc_calcWeight((cocirc_params*) params, prePos, postPos, &weight);

         // Just sum all connections coming into a given postsynaptic neuron:
         phi[i] += weight;
      } // for each input neuron
   }
   return 0;
}
