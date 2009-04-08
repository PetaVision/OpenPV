/*
 * PoolConn.cpp
 *
 *  Created on: Apr 7, 2009
 *      Author: rasmussn
 */

#include "PoolConn.hpp"
#include <assert.h>
#include <string.h>

namespace PV {

PoolConn::PoolConn(const char * name,
                   HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel)
{
   this->connId = hc->numberOfConnections();
   this->name   = strdup(name);
   this->parent = hc;

   this->numBundles = 1;

   initialize(NULL, pre, post, channel);

   hc->addConnection(this);
}

int PoolConn::initializeWeights(const char * filename)
{
   if (filename == NULL) {
      PVParams * params = parent->parameters();
      const float strength = params->value(name, "strength");

      const int xScale = pre->clayer->xScale;
      const int yScale = pre->clayer->yScale;

      int nfPre = pre->clayer->numFeatures;

      const int numPatches = numberOfWeightPatches();
      for (int i = 0; i < numPatches; i++) {
         int fPre = i % nfPre;
         poolWeights(wPatches[i], fPre, xScale, yScale, strength);
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

   // get parameters

   PVParams * params = parent->parameters();

   const int nx = (int) wp->nx;
   const int ny = (int) wp->ny;
   const int nf = (int) wp->nf;

   // strides
   const int sx = (int) wp->sx;  assert(sx == nf);
   const int sy = (int) wp->sy;  assert(sy == nf*nx);
   const int sf = (int) wp->sf;  assert(sf == 1);

   const float dx = powf(2, xScale);
   const float dy = powf(2, yScale);

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
