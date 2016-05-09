/*
 * LIF.cpp
 *
 *  Created on: Aug 4, 2008
 *      Author: dcoates
 */

// ---------------------------------------------------
//  Common leaky integrate-and-fire layer routines
// ---------------------------------------------------

#include "LIF.h"
#include "../include/pv_common.h"
#include "../utils/pv_random.h"

#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#ifdef _MSC_VER
#define inline _inline
#endif

#define PARAMS(p) (((LIF_params*)l->params)->p)

// Default handlers for a layer of leaky integrate-and-fire neurons.
static inline int update_f(PVLayer *l)
{
   int i;
   const float Vth = PARAMS(LIF_V_TH_0);

   for (i = 0; i < l->numNeurons; i++) {
      l->activity->data[i] = ((l->V[i] - Vth) > 0.0) ? 1.0 : 0.0;
      l->V[i] -= l->activity->data[i] * l->V[i]; // reset cells that fired
   }

   return 0;
}

int LIF_update(PVLayer *l)
{
   int i;
   float r = 0.0;
   float phiAve = 0.0, phiMax = FLT_MIN, phiMin = FLT_MAX;
   float VAve = 0.0, VMax = FLT_MIN, VMin = FLT_MAX;

   for (i = 0; i < l->numNeurons; i++) {
      if (pv_random_prob() < PARAMS(LIF_NOISE_FREQ)) {
         r = PARAMS(LIF_NOISE_AMP) * 2 * (pv_random_prob() - 0.5);
      }
      else r = 0.0;

      l->V[i] += PARAMS(LIF_DT_d_TAU) * (r + l->phi[PHI0][i] - l->V[i]);
      l->V[i] = (l->V[i] < PARAMS(LIF_V_Min)) ? PARAMS(LIF_V_Min) : l->V[i]; // hard lower limit

      // Gather some statistics
      // TODO - take into account extended border
      phiAve += l->phi[0][i];
      if (l->phi[PHI0][i] < phiMin) phiMin = l->phi[PHI0][i];
      if (l->phi[PHI0][i] > phiMax) phiMax = l->phi[PHI0][i];

      VAve += l->V[i];
      if (l->V[i] < VMin) VMin = l->V[i];
      if (l->V[i] > VMax) VMax = l->V[i];

      //pv_log("V1 PHI %d=%f\n", i, phi[i]);

      // TODO - take into account extended border
      l->phi[PHI0][i] = 0.0;
   }

#ifdef DEBUG_OUTPUT
   {
      char msg[128];
      sprintf(msg, "IF1: phi: Max: %1.4f, Avg=%1.4f Min=%1.4f\n", phiMax, phiAve
            / l->numNeurons, phiMin);
      pv_log(stderr, msg);
      sprintf(msg, "IF1: V  : Max: %1.4f, Avg=%1.4f Min=%1.4f\n", VMax,
            VAve / l->numNeurons, VMin);
      pv_log(stderr, msg);
   }
#endif

   update_f(l);

   return 0;
}
