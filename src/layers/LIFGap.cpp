/*
 * LIFGap.cpp
 *
 *  Created on: Jul 29, 2011
 *      Author: garkenyon
 */

#include "LIFGap.hpp"
#include "components/LIFGapActivityComponent.hpp"

namespace PV {

LIFGap::LIFGap(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

LIFGap::LIFGap() {}

LIFGap::~LIFGap() {}

void LIFGap::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   LIF::initialize(params, defaults, comm);
}

ActivityComponent *LIFGap::createActivityComponent() {
   return new LIFGapActivityComponent(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end namespace PV
