/*
 * ANNSquaredLayer.cpp
 *
 *  Created on: Sep 21, 2011
 *      Author: kpeterson
 */

#include "ANNSquaredLayer.hpp"
#include "components/SquaredInternalStateBuffer.hpp"

namespace PV {

ANNSquaredLayer::ANNSquaredLayer() { initialize_base(); }

ANNSquaredLayer::ANNSquaredLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

ANNSquaredLayer::~ANNSquaredLayer() {}

int ANNSquaredLayer::initialize_base() { return PV_SUCCESS; }

int ANNSquaredLayer::initialize(const char *name, HyPerCol *hc) {
   int status = ANNLayer::initialize(name, hc);
   return status;
}

InternalStateBuffer *ANNSquaredLayer::createInternalState() {
   return new SquaredInternalStateBuffer(getName(), parent);
}

Response::Status ANNSquaredLayer::allocateDataStructures() {
   auto status = ANNLayer::allocateDataStructures();
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

} /* namespace PV */
