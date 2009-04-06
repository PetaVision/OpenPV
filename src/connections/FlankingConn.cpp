/*
 * FileConn.cpp
 *
 *  Created on: Oct 27, 2008
 *      Author: rasmussn
 */

#include "FlankingConn.hpp"
#include "../io/io.h"
#include <assert.h>
#include <string.h>

namespace PV {

FlankingConn::FlankingConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post)
{
   this->connId = hc->numberOfConnections();
   this->name = strdup(name);
   this->parent = hc;
   this->numBundles = 1;

   initialize(NULL, pre, post, CHANNEL_EXC);

   hc->addConnection(this);
}

int FlankingConn::initializeWeights(const char * filename)
{
   PVParams * params = parent->parameters();

   float strength = 1.0;

   const float aspect = params->value(name, "aspect");
   const float sigma  = params->value(name, "sigma");
   const float rMax   = params->value(name, "rMax");
   if (params->present(name, "gaussWeightScale")) {
      strength = params->value(name, "gaussWeightScale");
   }

   float r2Max = rMax * rMax;

   int numFlanks = 1;
   float shift  = 0.0;
   float rotate = 1.0;

   if (params->present(name, "rotate")) rotate = params->value(name, "rotate");
   if (params->present(name, "numFlanks"))  numFlanks = params->value(name, "numFlanks");
   if (params->present(name, "flankShift")) shift     = params->value(name, "flankShift");

   int nfPre = pre->clayer->numFeatures;

   const int numPatches = numberOfWeightPatches();
   for (int i = 0; i < numPatches; i++) {
      int xScale = post->clayer->xScale - pre->clayer->xScale;
      int yScale = post->clayer->xScale - pre->clayer->yScale;
      int fPre = i % nfPre;
      gauss2DCalcWeights(wPatches[i], fPre, xScale, yScale,
                         numFlanks, shift, rotate, aspect, sigma, r2Max, strength);
   }

   return 0;
}

}
