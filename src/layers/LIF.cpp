/*
 * LIF.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: Craig Rasmussen
 */

#include "LIF.hpp"
#include "components/LIFActivityComponent.hpp"

namespace PV {

LIF::LIF(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

LIF::LIF() {}

LIF::~LIF() {}

void LIF::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

ActivityComponent *LIF::createActivityComponent() {
   return new LIFActivityComponent(mParamsIO, mCommunicator);
}

} // end namespace PV
