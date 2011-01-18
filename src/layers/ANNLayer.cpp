/*
 * NonspikingLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#include "ANNLayer.hpp"

namespace PV {
ANNLayer::ANNLayer(const char* name, HyPerCol * hc) : HyPerLayer(name, hc, MAX_CHANNELS) {
    initialize();
}  // end NonspikingLayer::NonspikingLayer()

ANNLayer::~ANNLayer() {}

int ANNLayer::initialize() {
    return HyPerLayer::initialize(TypeNonspiking);
}

}  // end namespace PV
