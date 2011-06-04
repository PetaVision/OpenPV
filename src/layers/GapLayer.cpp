/*
 * CloneLayer.cpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#include "HyPerLayer.hpp"
#include "GapLayer.hpp"

// CloneLayer can be used to implement gap junctions
namespace PV {
GapLayer::GapLayer(const char * name, HyPerCol * hc, LIF * originalLayer) :
   HyPerLayer(name, hc, MAX_CHANNELS)
{
   initialize(originalLayer);
}

GapLayer::~GapLayer()
{
    clayer->V = NULL;
}

int GapLayer::initialize(LIF * originalLayer)
{
   int status_init = HyPerLayer::initialize(TypeNonspiking);
   this->spikingFlag = false;
   sourceLayer = originalLayer;
   free(clayer->V);
   clayer->V = sourceLayer->getV();
   return status_init;
}

int GapLayer::updateV() {
   pvdata_t * V = getV();
   pvdata_t * GSynExc = getChannel(CHANNEL_EXC);
   pvdata_t exp_deltaT = 1.0f - exp(-this->getParent()->getDeltaTime() / sourceLayer->getLIFParams()->tau);
   for( int k=0; k<getNumNeurons(); k++ ) {
//      V[k] += GSynExc[k];  // different from superclass behavior, adds to V rather than replacing
      V[k] += GSynExc[k] * exp_deltaT;  //!!! uses base tau, not the true time-dep tau
   }
   return PV_SUCCESS;
}

int GapLayer::setActivity() {

   HyPerLayer::setActivity();

   // extended activity may not be current but this is alright since only local activity is used
   // !!! will break (non-deterministic) if layers are updated simultaneously--fix is to use datastore
   if (sourceLayer->getSpikingFlag()) { // probably not needed since numActive will be zero for non-spiking
      pvdata_t * sourceActivity = sourceLayer->getCLayer()->activity->data;
      pvdata_t * localActivity = getCLayer()->activity->data;
      unsigned int * activeNdx = sourceLayer->getCLayer()->activeIndices;
      for (unsigned int kActive = 0; kActive < sourceLayer->getCLayer()->numActive; kActive++) {
         int kex = activeNdx[kActive];
         localActivity[kex] += 50 * sourceActivity[kex]; // add 50 mV spike to local membrane potential
      }
   }
   return PV_SUCCESS;
}


} // end namespace PV

