/*
 * ComparisonLayer.cpp
 * Author: slundquist
 */

#include "ComparisonLayer.hpp"
#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>

namespace PV {
ComparisonLayer::ComparisonLayer(const char * name, HyPerCol * hc)
{
   initialize(name, hc);
}

int ComparisonLayer::updateState(double timef, double dt) {
   ANNLayer::updateState(timef, dt);
   int numNeurons = getNumNeurons();
   pvdata_t * A = getCLayer()->activity->data;
   const PVLayerLoc * loc = getLayerLoc(); 
   for(int ni = 0; ni < getNumNeurons(); ni++){
      int nExt = kIndexExtended(ni, loc->nx, loc->ny, loc->nf, loc->nb);
      //.1 is the error allowed
      assert(fabs(A[nExt]) <= .1);
   }
   return PV_SUCCESS;
}

}
