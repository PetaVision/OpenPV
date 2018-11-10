/*
 * StrengthParam.cpp
 *
 *  Created on: Jan 29, 2018
 *      Author: Pete Schultz
 */

#include "StrengthParam.hpp"

namespace PV {

StrengthParam::StrengthParam(char const *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

StrengthParam::~StrengthParam() {}

void StrengthParam::initialize(char const *name, PVParams *params, Communicator *comm) {
   BaseObject::initialize(name, params, comm);
}

void StrengthParam::setObjectType() { mObjectType = "StrengthParam"; }

int StrengthParam::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_strength(ioFlag);
   return PV_SUCCESS;
}

void StrengthParam::ioParam_strength(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "strength", &mStrength, mStrength);
}

} // namespace PV
