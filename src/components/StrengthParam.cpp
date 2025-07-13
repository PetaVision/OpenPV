/*
 * StrengthParam.cpp
 *
 *  Created on: Jan 29, 2018
 *      Author: Pete Schultz
 */

#include "StrengthParam.hpp"

namespace PV {

StrengthParam::StrengthParam(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

StrengthParam::~StrengthParam() {}

void StrengthParam::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   BaseObject::initialize(params, defaults, comm);
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
