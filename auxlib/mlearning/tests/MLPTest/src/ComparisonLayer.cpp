/*
 * ComparisonLayer.cpp
 * Author: slundquist
 */

#include "ComparisonLayer.hpp"
#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>

namespace PVMLearning{
ComparisonLayer::ComparisonLayer(const char * name, PV::HyPerCol * hc)
{
   initialize(name, hc);
}

int ComparisonLayer::updateState(double timef, double dt) {
   ANNLayer::updateState(timef, dt);
   int numNeurons = getNumNeurons();
   pvdata_t * A = getCLayer()->activity->data;

   pvdata_t * GSynExt = getChannel(CHANNEL_EXC); //gt
   pvdata_t * GSynInh = getChannel(CHANNEL_INH); //guess

   const PVLayerLoc * loc = getLayerLoc(); 
   float thresh = .5;
   for(int ni = 0; ni < getNumNeurons(); ni++){
      float guess = GSynInh[ni] <= thresh ? 0:1;
      float actual = GSynExt[ni];
      //std::cout << "guess: (" << GSynInh[ni] << "," << guess << ") actual: (" << GSynExt[ni] << "," << actual << ")\n" ;
      assert(guess == actual);
   }
   return PV_SUCCESS;
}

PV::BaseObject * createComparisonLayer(char const * name, PV::HyPerCol * hc) {
   return hc ? new ComparisonLayer(name, hc) : NULL;
}

}  // end namespace PVMLearning
