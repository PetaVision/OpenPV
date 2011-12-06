/*
 * ANNLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#include "ANNLayer.hpp"

namespace PV {

ANNLayer::ANNLayer() {
   initialize_base();
}

ANNLayer::ANNLayer(const char * name, HyPerCol * hc, int numChannels) {
   initialize_base();
   initialize(name, hc, numChannels);
}  // end ANNLayer::ANNLayer(const char *, HyPerCol *)

ANNLayer::~ANNLayer() {}

int ANNLayer::initialize_base() {
   return PV_SUCCESS;
}

int ANNLayer::initialize(const char * name, HyPerCol * hc, int numChannels) {
   int status = HyPerLayer::initialize(name, hc, numChannels);
   assert(status == PV_SUCCESS);
   PVParams * params = parent->parameters();

   return readVThreshParams(params);
}

int ANNLayer::readVThreshParams(PVParams * params) {
   VMax = params->value(name, "VMax", max_pvdata_t);
   VThresh = params->value(name, "VThresh", -max_pvdata_t);
   VMin = params->value(name, "VMin", VThresh);
   return PV_SUCCESS;
}

int ANNLayer::updateV() {
   HyPerLayer::updateV();
   applyVMax();
   applyVThresh();
   return PV_SUCCESS;
}

int ANNLayer::applyVMax() {
   if( VMax < FLT_MAX ) {
      pvdata_t * V = getV();
      for( int k=0; k<getNumNeurons(); k++ ) {
         if(V[k] > VMax) V[k] = VMax;
      }
   }
   return PV_SUCCESS;
}

int ANNLayer::applyVThresh() {
   if( VThresh > -FLT_MIN ) {
      pvdata_t * V = getV();
      for( int k=0; k<getNumNeurons(); k++ ) {
         if(V[k] < VThresh)
            V[k] = VMin;
      }
   }
   return PV_SUCCESS;
}


}  // end namespace PV
