/*
 * ANNSquaredLayer.cpp
 *
 *  Created on: Sep 21, 2011
 *      Author: kpeterson
 */

#include "ANNSquaredLayer.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/ActivityComponentWithInternalState.hpp"
#include "components/SquaredInternalStateBuffer.hpp"

namespace PV {

ANNSquaredLayer::ANNSquaredLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

ANNSquaredLayer::~ANNSquaredLayer() {}

int ANNSquaredLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

ActivityComponent *ANNSquaredLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<SquaredInternalStateBuffer, ANNActivityBuffer>(
         getName(), parent);
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
