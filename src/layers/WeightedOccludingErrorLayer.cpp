/*
 * WeightedOccludingErrorLayer.cpp
 *
 *  Created on: Jul 19, 2019
 *      Author: Jacob Springer
 */

#include "WeightedOccludingErrorLayer.hpp"
#include "components/HyPerActivityBuffer.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/HyPerActivityComponent.hpp"
#include "components/WeightedOccludingInternalStateBuffer.hpp"
#include "components/LayerInputBuffer.hpp"

namespace PV {

WeightedOccludingErrorLayer::WeightedOccludingErrorLayer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

WeightedOccludingErrorLayer::~WeightedOccludingErrorLayer() {}

void WeightedOccludingErrorLayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

LayerInputBuffer *WeightedOccludingErrorLayer::createLayerInput() {
   return new LayerInputBuffer(name, parameters(), mCommunicator);
}

ActivityComponent *WeightedOccludingErrorLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     WeightedOccludingInternalStateBuffer,
                                     HyPerActivityBuffer>(getName(), parameters(), mCommunicator);
}

} // end namespace PV
