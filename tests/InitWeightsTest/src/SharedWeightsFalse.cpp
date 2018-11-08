/*
 * SharedWeightsFalse.cpp
 *
 *  Created on: Jan 8, 2018
 *      Author: Pete Schultz
 */

#include "SharedWeightsFalse.hpp"
#include "columns/HyPerCol.hpp"
#include "components/ConnectionData.hpp"

namespace PV {

SharedWeightsFalse::SharedWeightsFalse(char const *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

SharedWeightsFalse::~SharedWeightsFalse() {}

void SharedWeightsFalse::initialize(char const *name, PVParams *params, Communicator *comm) {
   SharedWeights::initialize(name, params, comm);
}

void SharedWeightsFalse::setObjectType() { mObjectType = "SharedWeightsFalse"; }

int SharedWeightsFalse::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   return SharedWeights::ioParamsFillGroup(ioFlag);
}
void SharedWeightsFalse::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      mSharedWeights = false;
      parameters()->handleUnnecessaryParameter(name, "sharedWeights", mSharedWeights);
   }
}

} // namespace PV
