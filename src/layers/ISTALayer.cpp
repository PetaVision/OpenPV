/*
 * ISTALayer.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#include "ISTALayer.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/HyPerActivityComponent.hpp"
#include "components/ISTAInternalStateBuffer.hpp"
#include "components/LayerInputBuffer.hpp"

namespace PV {

ISTALayer::ISTALayer(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

ISTALayer::~ISTALayer() {}

void ISTALayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

LayerInputBuffer *ISTALayer::createLayerInput() {
   return new LayerInputBuffer(name, parameters(), mCommunicator);
}

ActivityComponent *ISTALayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator, ISTAInternalStateBuffer, ANNActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
