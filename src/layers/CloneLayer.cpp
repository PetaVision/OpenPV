/*
 * CloneLayer.cpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#include "HyPerLayer.hpp"
#include "CloneLayer.hpp"

// CloneLayer can be used to implement gap junctions
namespace PV {
CloneLayer::CloneLayer(const char * name, HyPerCol * hc, HyPerLayer * originalLayer) :
   HyPerLayer(name, hc, MAX_CHANNELS)
{
   initialize(sourceLayer);
}

CloneLayer::~CloneLayer()
{
    clayer->V = NULL;
}

int CloneLayer::initialize(HyPerLayer * originalLayer)
{
   int status_init = HyPerLayer::initialize(TypeNonspiking);
   sourceLayer = originalLayer;
   free(clayer->V);
   clayer->V = sourceLayer->getV();
  return status_init;
}

int CloneLayer::updateV() {
   pvdata_t * V = getV();
   pvdata_t * phiExc = getChannel(CHANNEL_EXC);
   for( int k=0; k<getNumNeurons(); k++ ) {
      V[k] += phiExc[k];  // different from superclass behavior, adds to V rather than replacing
   }
   return PV_SUCCESS;
}

int CloneLayer::setActivity() {
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

