/*
 * InhibConn.cpp
 *
 *  Created on: Feb 16, 2009
 *      Author: rasmussn
 */

#include "InhibConn.hpp"
#include "../io/io.h"
#include <assert.h>
#include <string.h>

namespace PV {

InhibConn::InhibConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post)
         : HyPerConn(name, hc, pre, post, CHANNEL_INH, PROTECTED_NUMBER)
{
   this->nfPre = pre->clayer->numFeatures;
   this->numAxonalArborLists = this->nfPre;
   initialize();
   hc->addConnection(this);
}

int InhibConn::initializeWeights(const char * filename)
{
   float aspect = 4.0;
   float sigma  = 2.0;
   float rMax   = 8.0;
   float lambda = sigma/0.8;    // gabor wavelength
   float strength = 1.0;

   PVParams * params = parent->parameters();

   aspect = params->value(name, "aspect");
   sigma  = params->value(name, "sigma");
   rMax   = params->value(name, "rMax");
   lambda = params->value(name, "lambda");

   if (params->present(name, "gaussWeightScale")) {
      strength = params->value(name, "gaussWeightScale");
   }

   int arbor = 0;
   const int numPatches = numberOfWeightPatches(arbor);
   for (int i = 0; i < numPatches; i++) {
      inhibWeights(wPatches[arbor][i], i, strength);
   }

   return 0;
}

/**
 * calculate gaussian weights to segment lines
 */
int InhibConn::inhibWeights(PVPatch * wp, int featureIndex, float strength)
{
   pvdata_t * w = wp->data;

   const int nx = (int) wp->nx;
   const int ny = (int) wp->ny;
   const int nf = (int) wp->nf;

   const int sx = (int) wp->sx;  assert(sx == nf);
   const int sy = (int) wp->sy;  assert(sy == nf*nx);
   const int sf = (int) wp->sf;  assert(sf == 1);

   // pre-synaptic neuron is at the center of the patch (0,0)
   // (x0,y0) is at upper left corner of patch (i=0,j=0)

   // weights zero except for matching feature
   for (int n = 0; n < nx*ny*nf; n++) {
      w[n] = 0.0;
   }

   int f = featureIndex;
   for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
         w[i*sx + j*sy + f*sf] = strength;
      }
   }

   // TODO - normalize?

   return 0;
}

} // namespace PV
