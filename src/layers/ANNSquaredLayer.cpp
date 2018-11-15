/*
 * ANNSquaredLayer.cpp
 *
 *  Created on: Sep 21, 2011
 *      Author: kpeterson
 */

#include "ANNSquaredLayer.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/HyPerActivityComponent.hpp"
#include "components/HyPerInternalStateBuffer.hpp"
#include "components/SquaredGSynAccumulator.hpp"

namespace PV {

ANNSquaredLayer::ANNSquaredLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

ANNSquaredLayer::~ANNSquaredLayer() {}

void ANNSquaredLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *ANNSquaredLayer::createActivityComponent() {
   return new HyPerActivityComponent<SquaredGSynAccumulator,
                                     HyPerInternalStateBuffer,
                                     ANNActivityBuffer>(getName(), parameters(), mCommunicator);
}

Response::Status ANNSquaredLayer::allocateDataStructures() {
   auto status = HyPerLayer::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   FatalIf(
         mLayerInput == nullptr or mLayerInput->getNumChannels() == 0,
         "%s requires a LayerInputBuffer component excitatory channel.\n",
         getDescription_c());
   if (mLayerInput->getNumChannels() != 1) {
      WarnLog() << getDescription()
                << " has more than one channel; only the excitatory channel will be used.\n";
   }
   return status;
}

} // end namespace PV
