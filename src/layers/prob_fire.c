/*
 * prob_fire.cpp
 *
 *  Created on: Aug 4, 2008
 *      Author: dcoates
 */

// ---------------------------------------------------
//  Fire probabilistically based on 'weight' of inputs
// ---------------------------------------------------

#include "prob_fire.h"
#include "../include/pv_common.h"

#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#define PARAMS(p) (((probFire_params*)l->params)->p)
void probFire_update(PVLayer *l)
{
   // For this type of layer, we don't care about V.
   // Just look at phi each timestep and fire accordingly.
   const float INV_RAND_MAX = 1.0 / (float) RAND_MAX;

   int i;
   float prob;
   for (i = 0; i < l->numNeurons; i++) {
      l->activity->data[i] = 0.0;

      // TODO - take into account extended border
      prob = (l->phi[PHI0][i] < PARAMS(PROB_FIRE_THRESHOLD)) ? PARAMS(PROB_FIRE_THRESHOLD) : l->phi[PHI0][i];
      prob *= PARAMS(PROB_FIRE_SCALE);

      if (rand() * INV_RAND_MAX < PARAMS(PROB_FIRE_NOISE_FREQ)) {
         prob += PARAMS(PROB_FIRE_NOISE_AMP) * 2 * (rand() * INV_RAND_MAX - 0.5);
      }

      if (rand() < prob * RAND_MAX) l->activity->data[i] = PARAMS(PROB_FIRE_AMP);

      l->phi[PHI0][i] = 0.0;
   }
}
