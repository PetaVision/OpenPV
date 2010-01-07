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
