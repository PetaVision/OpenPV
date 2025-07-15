/*
 * StrengthParam.cpp
 *
 *  Created on: Jan 29, 2018
 *      Author: Pete Schultz
 */

#include "StrengthParam.hpp"

namespace PV {

StrengthParam::StrengthParam(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

StrengthParam::~StrengthParam() {}

void StrengthParam::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   BaseObject::initialize(paramsIO, comm);
}

void StrengthParam::setObjectType() { mObjectType = "StrengthParam"; }

int StrengthParam::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   ioParam_strength(ioSwitch);
   return PV_SUCCESS;
}

void StrengthParam::ioParam_strength(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "strength", &mStrength);
}

} // namespace PV
