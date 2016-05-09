/*
 * PoolConn.cpp
 *
 *  Created on: Apr 7, 2009
 *      Author: rasmussn
 */
#ifdef OBSOLETE // Use KernelConn or HyperConn and set the param "weightInitType" to "PoolWeight" in the params file

#include "PoolConn.hpp"
#include <assert.h>
#include <string.h>

namespace PV {

PoolConn::PoolConn(const char * name,
                   HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, NULL);
}

int PoolConn::initializeWeights(const char * filename)
{
   if (filename == NULL) {
      PVParams * params = parent->parameters();
      const float strength = params->value(name, "strength");

      const int xScale = pre->clayer->xScale;
      const int yScale = pre->clayer->yScale;

      int nfPre = pre->clayer->loc.nf;

      const int arbor = 0;
      const int numPatches = numWeightPatches(arbor);
      for (int i = 0; i < numPatches; i++) {
         int fPre = i % nfPre;
         poolWeights(wPatches[arbor][i], fPre, xScale, yScale, strength);
      }
   }
   else {
      fprintf(stderr, "Initializing weights from a file not implemented for RuleConn\n");
      exit(1);
   } // end if for filename

   return 0;
}

int PoolConn::poolWeights(PVPatch * wp, int fPre, int xScale, int yScale, float strength)
{
   pvdata_t * w = wp->data;

   const int nx = wp->nx;
   const int ny = wp->ny;
   const int nf = wp->nf;

   // strides
   const int sx = wp->sx;  assert(sx == nf);
   const int sy = wp->sy;  assert(sy == nf*nx);
   const int sf = wp->sf;  assert(sf == 1);

   assert(fPre >= 0 && fPre <= 15);
   assert(nx == 1);
   assert(ny == 1);
   assert(nf == 2);

   // initialize connections of OFF and ON cells to 0
   for (int f = 0; f < nf; f++) {
      for (int j = 0; j < ny; j++) {
         for (int i = 0; i < nx; i++) {
            w[i*sx + j*sy + f*sf] = 0;
         }
      }
   }

   // connect an OFF cells to all OFF cells (and vice versa)

   for (int f = (fPre % 2); f < nf; f += 2) {
      w[0*sx + 0*sy + f*sf] = 1;
   }

   for (int f = 0; f < nf; f++) {
      float factor = strength;
      for (int i = 0; i < nx*ny; i++) w[f + i*nf] *= factor;
   }

   return 0;
}

} // namespace PV
#endif
