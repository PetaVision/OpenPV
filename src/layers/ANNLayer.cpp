/*
 * NonspikingLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#include "ANNLayer.hpp"

namespace PV {
ANNLayer::ANNLayer(const char * name, HyPerCol * hc) : HyPerLayer(name, hc, MAX_CHANNELS) {
   initialize();
}  // end ANNLayer::ANNLayer(const char *, HyPerCol *)

ANNLayer::~ANNLayer() {}

int ANNLayer::initialize() {
   HyPerLayer::initialize(TypeNonspiking);
   PVParams * params = parent->parameters();

   // moved to a separate routine so that derived classes that don't use
   // VThresh, VMax, VMin don't have to read them.
   return readVThreshParams(params);
}

int ANNLayer::readVThreshParams(PVParams * params) {
   VThresh = params->value(name, "VThresh", -max_pvdata_t);
   VMax = params->value(name, "VMax", max_pvdata_t);
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
         if(V[k] < VThresh) V[k] = VMin;
      }
   }
   return PV_SUCCESS;
}


}  // end namespace PV
