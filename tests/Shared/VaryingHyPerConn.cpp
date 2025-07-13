/*
 * VaryingHyPerConn.cpp
 *
 *  Created on: Nov 10, 2011
 *      Author: pschultz
 */

#include "VaryingHyPerConn.hpp"
#include "IncrementingWeightUpdater.hpp"

namespace PV {

VaryingHyPerConn::VaryingHyPerConn(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm)
      : HyPerConn() {
   initialize(params, defaults, comm);
}

VaryingHyPerConn::~VaryingHyPerConn() {}

void VaryingHyPerConn::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerConn::initialize(params, defaults, comm);
}

BaseWeightUpdater *VaryingHyPerConn::createWeightUpdater() {
   return new IncrementingWeightUpdater(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end of namespace PV block
