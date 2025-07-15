/*
 * LIFGap.cpp
 *
 *  Created on: Jul 29, 2011
 *      Author: garkenyon
 */

#include "LIFGap.hpp"
#include "components/LIFGapActivityComponent.hpp"

namespace PV {

LIFGap::LIFGap(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

LIFGap::LIFGap() {}

LIFGap::~LIFGap() {}

void LIFGap::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   LIF::initialize(paramsIO, comm);
}

ActivityComponent *LIFGap::createActivityComponent() {
   return new LIFGapActivityComponent(mParamsIO, mCommunicator);
}

} // end namespace PV
