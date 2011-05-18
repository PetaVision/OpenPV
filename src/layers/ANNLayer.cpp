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
    VThresh = params->value(name, "VThresh", -max_pvdata_t);
    VMax = params->value(name, "VMax", max_pvdata_t);
    VMin = params->value(name, "VMin", VThresh);
    return PV_SUCCESS;
}

int ANNLayer::updateV() {
   HyPerLayer::updateV();
   pvdata_t * V = getV();
   for( int k=0; k<getNumNeurons(); k++ ) {
     V[k] = V[k] > VMax ? VMax : V[k];
     V[k] = V[k] < VThresh ? VMin : V[k];
   }
   return PV_SUCCESS;
}


}  // end namespace PV
