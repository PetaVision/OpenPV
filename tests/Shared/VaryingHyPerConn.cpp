/*
 * VaryingHyPerConn.cpp
 *
 *  Created on: Nov 10, 2011
 *      Author: pschultz
 */

#include "VaryingHyPerConn.hpp"
#include "IncrementingWeightUpdater.hpp"

namespace PV {

VaryingHyPerConn::VaryingHyPerConn(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm)
      : HyPerConn() {
   initialize(paramsIO, comm);
}

VaryingHyPerConn::~VaryingHyPerConn() {}

void VaryingHyPerConn::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   HyPerConn::initialize(paramsIO, comm);
}

BaseWeightUpdater *VaryingHyPerConn::createWeightUpdater() {
   return new IncrementingWeightUpdater(mParamsIO, mCommunicator);
}

} // end of namespace PV block
