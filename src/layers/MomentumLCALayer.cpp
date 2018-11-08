/*
 * MomentumLCALayer.cpp
 *
 *  Created on: Mar 15, 2016
 *      Author: slundquist
 */

#include "MomentumLCALayer.hpp"
#include "components/MomentumLCAActivityComponent.hpp"
#include "components/TauLayerInputBuffer.hpp"

namespace PV {

MomentumLCALayer::MomentumLCALayer(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

MomentumLCALayer::~MomentumLCALayer() {}

void MomentumLCALayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   HyPerLCALayer::initialize(name, params, comm);
}

ActivityComponent *MomentumLCALayer::createActivityComponent() {
   return new MomentumLCAActivityComponent(getName(), parameters(), mCommunicator);
}

} // end namespace PV
