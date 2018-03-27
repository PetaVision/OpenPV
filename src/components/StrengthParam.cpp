/*
 * StrengthParam.cpp
 *
 *  Created on: Jan 29, 2018
 *      Author: Pete Schultz
 */

#include "StrengthParam.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

StrengthParam::StrengthParam(char const *name, HyPerCol *hc) { initialize(name, hc); }

StrengthParam::~StrengthParam() {}

int StrengthParam::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

void StrengthParam::setObjectType() { mObjectType = "StrengthParam"; }

int StrengthParam::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_strength(ioFlag);
   return PV_SUCCESS;
}

void StrengthParam::ioParam_strength(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "strength", &mStrength, mStrength);
}

} // namespace PV
