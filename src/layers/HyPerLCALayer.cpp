/*
 * HyPerLCALayer.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#include "HyPerLCALayer.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/HyPerActivityComponent.hpp"
#include "components/HyPerLCAInternalStateBuffer.hpp"
#include "components/LayerInputBuffer.hpp"

namespace PV {

HyPerLCALayer::HyPerLCALayer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

HyPerLCALayer::~HyPerLCALayer() {}

void HyPerLCALayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

LayerInputBuffer *HyPerLCALayer::createLayerInput() {
   return new LayerInputBuffer(getName(), parameters(), mCommunicator);
}

ActivityComponent *HyPerLCALayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     HyPerLCAInternalStateBuffer,
                                     ANNActivityBuffer>(getName(), parameters(), mCommunicator);
}

} // end namespace PV
