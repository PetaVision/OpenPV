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
int zPatchHead(int kzPre, int nzPatch, int zScaleLog2Pre, int zScaleLog2Post)
{
   int shift;

   float a = powf(2.0f, (float) (zScaleLog2Pre - zScaleLog2Post));

   if ((int) a == 1) {
      shift = - (int) (0.5f * (float) nzPatch);
      return shift + nearby_neighbor(kzPre, zScaleLog2Pre, zScaleLog2Post);
   }

   shift = 1 - (int) (0.5f * (float) nzPatch);

   if (nzPatch % 2 == 0 && a < 1) {
      // density increases in post-synaptic layer

      // extra shift subtracted if kzPre is in right half of the
      // set of presynaptic indices that are between postsynaptic
      //

      int kpos = (kzPre < 0) ? -(1+kzPre) : kzPre;
      int l = (int) (2*a*kpos) % 2;
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
      shift = - (int) (0.5f * (float) nzPatch);
   }


   int neighbor = nearby_neighbor(kzPre, zScaleLog2Pre, zScaleLog2Post);

   //added if nzPatch == 1
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
   assert(kzPre >= 0);  // not valid in general if kzPre < 0
   int kzPost = kzPre;
   *distPost = 0.0;
   *distPre = 0.0;
   if (log2ScalePre > log2ScalePost) {
      // post-synaptic layer has smaller size scale (is denser)
      int scaleFac = pow(2, log2ScalePre) / pow(2, log2ScalePost);
      *distPost = 0.5;
      *distPre = 0.5 / scaleFac;
      kzPost = (int) ((kzPre + 0.5) * scaleFac) - 1;  // left neighbor, add 1 for right neighbor
   }
   else if (log2ScalePre < log2ScalePost) {
      // post-synaptic layer has larger size scale (is less dense), scaleFac > 1
      int scaleFac = pow(2, log2ScalePost) / pow(2, log2ScalePre);
      *distPre = 0.5 * (scaleFac - 2 * (kzPre % scaleFac) - 1);
      *distPost = *distPre / scaleFac;
      kzPost = kzPre / scaleFac;
   }
   return kzPre;
}

#define DEPRECATED_FEATURES
#ifdef DEPRECATED_FEATURES
// deprecated
/*
 * returns global x,y position of patchhead and of presynaptic cell
 * requires kPre >= 0 in restricted space
 */
int posPatchHead(const int kPre, const int xScaleLog2Pre,
      const int yScaleLog2Pre, const PVLayerLoc locPre, float * xPreGlobal,
      float * yPreGlobal, const int xScaleLog2Post, const int yScaleLog2Post,
      const PVLayerLoc locPost, const PVPatch * wp, float * xPatchHeadGlobal,
      float * yPatchHeadGlobal)
{
   // get global index and location of presynaptic cell
   const int nxPre = locPre.nx;
   const int nyPre = locPre.ny;
   const int nfPre = locPre.nBands;
   const int nxPreGlobal = locPre.nxGlobal;
   const int nyPreGlobal = locPre.nyGlobal;
   const int kPreGlobal = globalIndexFromLocal(kPre, locPre, nfPre);
   *xPreGlobal = xPosGlobal(kPreGlobal, xScaleLog2Pre, nxPreGlobal,
         nyPreGlobal, nfPre);
   *yPreGlobal = yPosGlobal(kPreGlobal, yScaleLog2Pre, nxPreGlobal,
         nyPreGlobal, nfPre);

   // get global index of postsynaptic patchhead
   const int kxPre = (int) kxPos(kPre, nxPre, nyPre, nfPre);
   const int kyPre = (int) kyPos(kPre, nxPre, nyPre, nfPre);
   const int kxPatchHead = zPatchHead(kxPre, wp->nx, xScaleLog2Pre, xScaleLog2Post);
   const int kyPatchHead = zPatchHead(kyPre, wp->ny, yScaleLog2Pre, yScaleLog2Post);
   const int nxPost = locPost.nx;
   const int nyPost = locPost.ny;
   const int nfPost = locPost.nBands;
   const int kPatchHead = kIndex(kxPatchHead, kyPatchHead, 0, nxPost, nyPost, nfPost);
   const int kPatchHeadGlobal = globalIndexFromLocal(kPatchHead, locPost,
         nfPost);

   // get global x,y position of patchhead
   const float nxPostGlobal = locPost.nxGlobal;
   const float nyPostGlobal = locPost.nyGlobal;
   *xPatchHeadGlobal = xPosGlobal(kPatchHeadGlobal, xScaleLog2Post,
         nxPostGlobal, nyPostGlobal, nfPost);
   *yPatchHeadGlobal = yPosGlobal(kPatchHeadGlobal, yScaleLog2Post,
         nxPostGlobal, nyPostGlobal, nfPost);

   return 0;
}
#endif /* DEPRECATED_FEATURES */
