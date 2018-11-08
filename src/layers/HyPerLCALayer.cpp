/*
 * HyPerLCALayer.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#include "HyPerLCALayer.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/ActivityComponentWithInternalState.hpp"
#include "components/HyPerLCAInternalStateBuffer.hpp"
#include "components/TauLayerInputBuffer.hpp"

namespace PV {

HyPerLCALayer::HyPerLCALayer(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

HyPerLCALayer::~HyPerLCALayer() {}

void HyPerLCALayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

LayerInputBuffer *HyPerLCALayer::createLayerInput() {
   return new TauLayerInputBuffer(name, parameters(), mCommunicator);
}

ActivityComponent *HyPerLCALayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<HyPerLCAInternalStateBuffer, ANNActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
