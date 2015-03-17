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

   pvdata_t * GSynExt = getChannel(CHANNEL_EXC); //guess
   pvdata_t * GSynInh = getChannel(CHANNEL_INH); //gt

   const PVLayerLoc * loc = getLayerLoc(); 
   float thresh = .5;
   for(int ni = 0; ni < getNumNeurons(); ni++){
      float guess = GSynExt[ni] <= thresh ? 0:1;
      float actual = GSynInh[ni];
      assert(guess == actual);

   }
   return PV_SUCCESS;
}

}
