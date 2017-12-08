/*
 * NoCheckpointConnectionData.cpp
 *
 *  Created on: Dec 7, 2017
 *      Author: pschultz
 */

#include "NoCheckpointConnectionData.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

NoCheckpointConnectionData::NoCheckpointConnectionData(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

NoCheckpointConnectionData::NoCheckpointConnectionData() {}

NoCheckpointConnectionData::~NoCheckpointConnectionData() {}

int NoCheckpointConnectionData::initialize(char const *name, HyPerCol *hc) {
   return ConnectionData::initialize(name, hc);
}

int NoCheckpointConnectionData::setDescription() {
   description = "NoCheckpointConnectionData \"";
   description += name;
   description += "\"";
   return PV_SUCCESS;
}

void NoCheckpointConnectionData::ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      mInitializeFromCheckpointFlag = false;
      parent->parameters()->handleUnnecessaryParameter(
            name, "initializeFromCheckpointFlag", false /*correctValue*/);
   };
}

} // namespace PV
