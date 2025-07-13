/*
 * GapLayer.cpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#include "GapLayer.hpp"
#include "components/CloneActivityComponent.hpp"
#include "components/CloneInternalStateBuffer.hpp"
#include "components/GapActivityBuffer.hpp"

namespace PV {

GapLayer::GapLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

GapLayer::~GapLayer() {}

void GapLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   CloneVLayer::initialize(params, defaults, comm);
}

ActivityComponent *GapLayer::createActivityComponent() {
   return new CloneActivityComponent<CloneInternalStateBuffer, GapActivityBuffer>(
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end namespace PV
