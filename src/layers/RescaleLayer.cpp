/*
 * RescaleLayer.cpp
 */

#include "RescaleLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/RescaleActivityBuffer.hpp"

namespace PV {

RescaleLayer::RescaleLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

RescaleLayer::RescaleLayer() {}

RescaleLayer::~RescaleLayer() {}

void RescaleLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   CloneVLayer::initialize(params, defaults, comm);
}

ActivityComponent *RescaleLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<RescaleActivityBuffer>(
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end namespace PV
