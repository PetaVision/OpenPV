/*
 * transformations.c
 *
 *  Created on: Jan 6, 2010
 *      Author: rasmussn
 */

#include "conversions.h"
#include "assert.h"

/**
 * Return the leading index in z direction (either x or y) of a patch in postsynaptic layer
 * @kzPre is the pre-synaptic index in z direction (can be either local or global)
 * @zScaleLog2Pre is log2 zScale (distance not number) of presynaptic layer
 * @zScaleLog2Post is log2 zScale (distance not number) of postsynaptic layer
 * @nzPatch is the size of patch in z direction
 *
 * kzPre is always in restricted space
 */
int zPatchHead(int kzPre, int nzPatch, int zLog2ScaleDiff) {
   int shift;

   float a = powf(2.0f, -(float)(zLog2ScaleDiff));

   if ((int)a == 1) {
      shift = -(int)(0.5f * (float)nzPatch);
      return shift + nearby_neighbor(kzPre, zLog2ScaleDiff);
   }

   shift = 1 - (int)(0.5f * (float)nzPatch);

   if (nzPatch % 2 == 0 && a < 1) {
      // density increases in post-synaptic layer

      // extra shift subtracted if kzPre is in right half of the
      // set of presynaptic indices that are between postsynaptic
      //

      int kpos = (kzPre < 0) ? -(1 + kzPre) : kzPre;
      int l    = (int)(2 * a * kpos) % 2;
      // The following statement performs this:
      // if (kzPre < 0) {
      //    if (l == 1) {
      //       shift -= 1
      //    }
      // }
      // else {
      //    if (l == 0) {
      //       shift -= 1
      //    }
      // }

      shift -= (kzPre < 0) ? l == 1 : l == 0;
   }
   else if (nzPatch % 2 == 1 && a < 1) {
      shift = -(int)(0.5f * (float)nzPatch);
   }

   int neighbor = nearby_neighbor(kzPre, zLog2ScaleDiff);

   // added if nzPatch == 1
   if (nzPatch == 1) {
      return neighbor;
   }

   return shift + neighbor;
}

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
