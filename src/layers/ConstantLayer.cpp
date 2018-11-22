/*
 * ConstantLayer.hpp
 *
 *  Created on: Dec 17, 2013
 *      Author: slundquist
 */

#include "ConstantLayer.hpp"

namespace PV {

ConstantLayer::ConstantLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

ConstantLayer::ConstantLayer() {}

ConstantLayer::~ConstantLayer() {}

void ConstantLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   // HyPerLayer default for writeStep is 1.0, but
   // writeStep = -1 (never write) is a better default for ConstantLayer.
   writeStep = -1;

   HyPerLayer::initialize(name, params, comm);
}

LayerUpdateController *ConstantLayer::createLayerUpdateController() { return nullptr; }

} /* namespace PV */
