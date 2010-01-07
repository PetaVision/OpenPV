/*
 * transformations.c
 *
 *  Created on: Jan 6, 2010
 *      Author: rasmussn
 */

#include "conversions.h"

/**
 * Return the leading index in z direction (either x or y) of a patch in postsynaptic layer
 * @kzPre is the pre-synaptic index in z direction (can be either local or global)
 * @zScaleLog2Pre is log2 zScale of presynaptic layer
 * @zScaleLog2Post is log2 zScale of postsynaptic layer
 * @nzPatch is the size of patch in z direction
 */
int zPatchHead(int kzPre, int nzPatch, int zScaleLog2Pre, int zScaleLog2Post)
{
   int shift = 0;

   if (nzPatch % 2 == 0 && (zScaleLog2Post < zScaleLog2Pre)) {
      // if even, can't shift evenly (at least for scale < 0)
      // the later choice alternates direction so not always to left
      shift = kzPre % 2;
   }
   shift -= (int) (0.5 * (float) nzPatch);
   return shift + nearby_neighbor(kzPre, zScaleLog2Pre, zScaleLog2Post);
}

/**
 * compute distance from kzPre to the nearest kzPost, i.e.
 *    (xPost - xPre) or (yPost - yPre)
 * in units of both pre- and post-synaptic dx (or dy).
 *
 * distance can be positive or negative
 *
 * also computes  kzPost, which is local x (or y) index of nearest cell in post layer
 *
 * @log2ScalePre
 * @log2ScalePost
 * @kzPost
 * @distPre
 * @distPost
 */
int dist2NearestCell(int kzPre, int log2ScalePre, int log2ScalePost,
      float * distPre, float * distPost)
{
   // scaleFac == 1
   int kzPost = kzPre;
   *distPost = 0.0;
   *distPre = 0.0;
   if (log2ScalePre > log2ScalePost) {
      // post-synaptic layer has smaller size scale (is denser)
      int scaleFac = pow(2, log2ScalePre) / pow(2, log2ScalePost);
      *distPost = 0.5;
      *distPre = 0.5 / scaleFac;
      kzPost = (int) ((kzPre + 0.5) * scaleFac) - 1;
   }
   else if (log2ScalePre < log2ScalePost) {
      // post-synaptic layer has larger size scale (is less dense), scaleFac > 1
      int scaleFac = pow(2, log2ScalePost) / pow(2, log2ScalePre);
      *distPre = 0.5 * (scaleFac - 2 * (kzPre % scaleFac) - 1);
      *distPost = *distPre / scaleFac;
   }
   return kzPre;
}

