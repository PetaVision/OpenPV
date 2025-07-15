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

GapLayer::GapLayer(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

GapLayer::~GapLayer() {}

void GapLayer::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   CloneVLayer::initialize(paramsIO, comm);
}

ActivityComponent *GapLayer::createActivityComponent() {
   return new CloneActivityComponent<CloneInternalStateBuffer, GapActivityBuffer>(
         mParamsIO, mCommunicator);
}

} // end namespace PV
