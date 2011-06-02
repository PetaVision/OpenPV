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

   // don't need conductance channels
   freeChannels();

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
   pvdata_t sig_scale = 1.0f;
   pvdata_t Vth = sourceLayer->getLIFParams()->VthRest;
   pvdata_t V0 = sourceLayer->getLIFParams()->Vrest;
   if ( Vth > V0 ){
      sig_scale = 1 / (Vth - V0);
   }
   for( int k=0; k<getNumNeurons(); k++ ) {
      int kex = kIndexExtended(k, nx, ny, nf, nb);
      if (V[k] > Vth){
         activity[kex] = 1.0f;
      }
      else if (V[k] < V0){
         activity[kex] = 0.0f;
      }
      else{
         activity[kex] = (V[k] - V0) * sig_scale;
      }
   }
   return PV_SUCCESS;
}


} // end namespace PV

