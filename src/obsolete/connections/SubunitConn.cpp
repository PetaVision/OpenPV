/*
 * LineSegments.cpp
 *
 *  Created on: Jan 26, 2009
 *      Author: rasmussn
 */
#ifdef OBSOLETE // Use KernelConn or HyperConn and set the param "weightInitType" to "SubUnitWeight" in the params file

#include "SubunitConn.hpp"
#include "../io/io.h"
#include <assert.h>
#include <string.h>

namespace PV {

/**
 * This connection is for retina to layer with 4 x 16 features.  The post-synaptic layer
 * exhaustively computes presence of a hierarchy of 4 x 2x2 (on/off) patch of pixels
 * (embedded in a 3x3 pixel patch).
 */
SubunitConn::SubunitConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
                         ChannelType channel)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, NULL);
}

/**
 * Initialize weights to all values of 4 x 2x2 patch
 */
int SubunitConn::initializeWeights(const char * filename)
{
   assert(post->clayer->loc.nf == 4*16);

   const int arbor = 0;
   const int numPatches = numWeightPatches(arbor);
   for (int i = 0; i < numPatches; i++) {
      weights(wPatches[arbor][i]);
   }

   return 0;
}

int SubunitConn::weights(PVPatch * wp)
{
   pvdata_t * w = wp->data;

   const int nx = wp->nx;
   const int ny = wp->ny;
   const int nf = wp->nf;

   const int sx = wp->sx;  assert(sx == nf);
   const int sy = wp->sy;  assert(sy == nf*nx);
   const int sf = wp->sf;  assert(sf == 1);

   assert(nx == 3);
   assert(ny == 3);
   assert(nf == 4*16);

   // TODO - already initialized to zero (so delete)
   for (int k = 0; k < nx*ny*nf; k++) {
      w[k] = 0.0;
   }

   for (int f = 0; f < nf; f++) {
      int i0 = 0, j0 = 0;
      int kf = f / 16;
      if (kf == 0) {i0 = 0; j0 = 0;}
      if (kf == 1) {i0 = 1; j0 = 0;}
      if (kf == 2) {i0 = 0; j0 = 1;}
      if (kf == 3) {i0 = 1; j0 = 1;}

      kf = f % 16;

      for (int j = 0; j < 2; j++) {
         for (int i = 0; i < 2; i++) {
            int n = i + 2*j;
            int r = kf >> n;
            r = 0x1 & r;
            w[(i+i0)*sx + (j+j0)*sy + f*sf] = r;
         }
      }
   }

   // normalize
   for (int f = 0; f < nf; f++) {
      float sum = 0;
      for (int i = 0; i < nx*ny; i++) sum += w[f + i*nf];

      if (sum == 0) continue;

      float factor = 1.0/sum;
      for (int i = 0; i < nx*ny; i++) w[f + i*nf] *= factor;
   }

   return 0;
}

} // namespace PV
#endif
