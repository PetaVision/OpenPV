/*
 * OccludingLayer.cpp
 *
 *  Created on: Jul 19, 2019
 *      Author: Jacob Springer
 */

#include "OccludingLayer.hpp"
#include "components/HyPerActivityBuffer.hpp"
#include "components/OccludingGSynAccumulator.hpp"
#include "components/HyPerActivityComponent.hpp"
#include "components/HyPerInternalStateBuffer.hpp"
#include "components/LayerInputBuffer.hpp"

namespace PV {

OccludingLayer::OccludingLayer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

OccludingLayer::~OccludingLayer() {}

void OccludingLayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

LayerInputBuffer *OccludingLayer::createLayerInput() {
   return new LayerInputBuffer(name, parameters(), mCommunicator);
}

ActivityComponent *OccludingLayer::createActivityComponent() {
   return new HyPerActivityComponent<OccludingGSynAccumulator,
                                     HyPerInternalStateBuffer,
                                     HyPerActivityBuffer>(getName(), parameters(), mCommunicator);
}

} // end namespace PV
