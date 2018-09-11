/*
 * InitializeFromCheckpointFlag.cpp
 *
 *  Created on: Jun 8, 2018
 *      Author: Pete Schultz
 */

#include "InitializeFromCheckpointFlag.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

InitializeFromCheckpointFlag::InitializeFromCheckpointFlag(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

InitializeFromCheckpointFlag::~InitializeFromCheckpointFlag() {}

int InitializeFromCheckpointFlag::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

void InitializeFromCheckpointFlag::setObjectType() { mObjectType = "InitializeFromCheckpointFlag"; }

int InitializeFromCheckpointFlag::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_initializeFromCheckpointFlag(ioFlag);
   return PV_SUCCESS;
}

void InitializeFromCheckpointFlag::ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag,
         name,
         "initializeFromCheckpointFlag",
         &mInitializeFromCheckpointFlag,
         mInitializeFromCheckpointFlag);
}

} // namespace PV
