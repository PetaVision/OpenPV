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

FlankingConn::FlankingConn(const char * name,
                           HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel)
            : HyPerConn(name, hc, pre, post, channel, PROTECTED_NUMBER)
{
   this->numAxonalArborLists = 1;
   initialize();
   hc->addConnection(this);
}

int FlankingConn::initializeWeights(const char * filename)
{
   PVParams * params = parent->parameters();

   float strength = 1.0;

   int noPost = 1;
   if (params->present(post->getName(), "no")) {
      noPost = params->value(post->getName(), "no");
   }

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

   const int borderId = 0;
   const int numPatches = numberOfWeightPatches(borderId);
   for (int i = 0; i < numPatches; i++) {
      int xScale = post->clayer->xScale - pre->clayer->xScale;
      int yScale = post->clayer->xScale - pre->clayer->yScale;
      int fPre = i % nfPre;
      gauss2DCalcWeights(wPatches[borderId][i], fPre, noPost, xScale, yScale,
                         numFlanks, shift, rotate, aspect, sigma, r2Max, strength);
   }

   return 0;
}

} // namespace PV
