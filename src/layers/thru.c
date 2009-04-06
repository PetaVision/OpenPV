/*
 * thru.c
 *
 *  Created on: Aug 10, 2008
 *      Author: dcoates
 */

#include "thru.h"
#include "../include/pv_common.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <memory.h>

#define PARAMS(p) (params->p)

int thru_rcv(PVLayer* pre, PVLayer* post, float *phi, int nActivity, float *fActivity,
             thru_params* params)
{
   int n, m;

#ifdef DEBUG_OUTPUT
   pv_log(stderr, "Thru: layer %d receiving %d neurons from layer %d\n", post->layerId,
          nActivity, pre->layerId);
#endif

   if (PARAMS(USE_F_DIRECTLY)) {
      if (PARAMS(DIM_EXPANSION) > 1) {
#ifdef DEBUG_OUTPUT
         pv_log(stderr, "Error: thru can't yet do direct dimension expansion!\n");
#endif
      }
      else {
         memcpy(post->activity->data, fActivity, post->numNeurons);
      }
   }

   // To receive FF input pass-through, just add it to our phi
   // loop for all incoming activity
   for (n = 0; n < nActivity; n++) {
      // TODO: we aren't sparse yet
      if (fActivity[n] == 0) continue;

      for (m = 0; m < PARAMS(DIM_EXPANSION); m++)
         // Just add the input to the phi
         // of the postsynaptic V1 neuron at the right place(s)
         phi[(int) (n * PARAMS(DIM_EXPANSION)) + m] += fActivity[n] * PARAMS(THRU_SCALE);

   }

   // Done for now, for feedforward connection.
   return 0;
}
