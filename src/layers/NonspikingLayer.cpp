/*
 * NonspikingLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#include "NonspikingLayer.hpp"

namespace PV {
NonspikingLayer::NonspikingLayer(const char* name, HyPerCol * hc) : HyPerLayer(name, hc) {
    initialize();
}  // end NonspikingLayer::NonspikingLayer()

NonspikingLayer::~NonspikingLayer() {}

int NonspikingLayer::initialize() {
    return HyPerLayer::initialize(TypeNonspiking);
}

}  // end namespace PV
