/*
 * transformations.c
 *
 *  Created on: Jan 6, 2010
 *      Author: rasmussn
 */

#include "conversions.h"

/**
 * compute distance from kzPre to the nearest kzPost, i.e.
 *    (xPost - xPre) or (yPost - yPre)
 * in units of both pre- and post-synaptic dx (or dy).
 *
 * distance can be positive or negative
 *
 * returns kzPost, which is local x (or y) index of nearest cell in post layer
 *
 * @log2ScalePre
 * @log2ScalePost
 * @kzPost
 * @distPre
 * @distPost
 */
int dist2NearestCell(int kzPre, int log2ScaleDiff, float *distPre, float *distPost) {
   // scaleFac == 1
   assert(kzPre >= 0); // not valid in general if kzPre < 0
   int kzPost = kzPre;
   *distPost  = 0.0;
   *distPre   = 0.0;
   if (log2ScaleDiff < 0) {
      // post-synaptic layer has smaller size scale (is denser)
      int scaleFac = pow(2, -log2ScaleDiff);
      *distPost    = -0.5;
      *distPre     = -0.5 / scaleFac;
      kzPost       = (int)((kzPre + 0.5) * scaleFac) - 1; // left neighbor, add 1 for right neighbor
   }
   else if (log2ScaleDiff > 0) {
      // post-synaptic layer has larger size scale (is less dense), scaleFac > 1
      int scaleFac = pow(2, log2ScaleDiff);
      *distPre     = 0.5 * (scaleFac - 2 * (kzPre % scaleFac) - 1);
      *distPost    = *distPre / scaleFac;
      kzPost       = kzPre / scaleFac;
   }
   return kzPost;
}
