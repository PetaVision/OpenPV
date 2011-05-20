/*
 * CloneLayer.cpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#include "HyPerLayer.hpp"
#include "SigmoidLayer.hpp"

// CloneLayer can be used to implement Sigmoid junctions
namespace PV {
SigmoidLayer::SigmoidLayer(const char * name, HyPerCol * hc, LIF * originalLayer) :
   HyPerLayer(name, hc, MAX_CHANNELS)
{
   initialize(originalLayer);
}

SigmoidLayer::~SigmoidLayer()
{
    clayer->V = NULL;
}

int SigmoidLayer::initialize(LIF * originalLayer)
{
   int status_init = HyPerLayer::initialize(TypeNonspiking);
   this->spikingFlag = false;
   sourceLayer = originalLayer;
   free(clayer->V);
   clayer->V = sourceLayer->getV();
   if (numChannels > 0) {
      // potentials allocated contiguously so this frees all
      free(phi[0]);
   }
   phi[0] = NULL;
   numChannels = 0;
   PVParams * params = parent->parameters();
   VThresh = params->value(name, "VThresh", -max_pvdata_t);
   VMax = params->value(name, "VMax", max_pvdata_t);
   VMin = params->value(name, "VMin", VThresh);
   return status_init;
}

int SigmoidLayer::updateV() {
   return PV_SUCCESS;
}

int SigmoidLayer::resetPhiBuffers() {
   return PV_SUCCESS;
}


int SigmoidLayer::setActivity() {

   const int nx = getLayerLoc()->nx;
   const int ny = getLayerLoc()->ny;
   const int nf = getLayerLoc()->nf;
   const int nb = getLayerLoc()->nb;
   pvdata_t * activity = getCLayer()->activity->data;
   pvdata_t * V = getV();
   for( int k=0; k<getNumExtended(); k++ ) {
      activity[k] = 0; // Would it be faster to only do the margins?
   }
   for( int k=0; k<getNumNeurons(); k++ ) {
      int kex = kIndexExtended(k, nx, ny, nf, nb);
      activity[kex] = V[k] > VMax ? VMax : V[k];
      activity[kex] = activity[kex] < VThresh ? VMin : activity[kex];
   }
   return PV_SUCCESS;
}


} // end namespace PV

