/*
 * RescaleLayer.cpp
 */

#include "RescaleLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/RescaleActivityBuffer.hpp"

namespace PV {

RescaleLayer::RescaleLayer(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

RescaleLayer::RescaleLayer() {}

RescaleLayer::~RescaleLayer() {}

void RescaleLayer::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   CloneVLayer::initialize(paramsIO, comm);
}

ActivityComponent *RescaleLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<RescaleActivityBuffer>(
         mParamsIO, mCommunicator);
}

} // end namespace PV
