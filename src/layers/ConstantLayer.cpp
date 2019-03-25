/*
 * ConstantLayer.hpp
 *
 *  Created on: Dec 17, 2013
 *      Author: slundquist
 */

#include "ConstantLayer.hpp"

#include "components/DefaultNoOutputComponent.hpp"

namespace PV {

ConstantLayer::ConstantLayer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

ConstantLayer::ConstantLayer() {}

ConstantLayer::~ConstantLayer() {}

void ConstantLayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

LayerUpdateController *ConstantLayer::createLayerUpdateController() { return nullptr; }

LayerOutputComponent *ConstantLayer::createLayerOutput() {
   return new DefaultNoOutputComponent(getName(), parameters(), mCommunicator);
}

} /* namespace PV */
