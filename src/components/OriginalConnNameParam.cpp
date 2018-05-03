/*
 * OriginalConnNameParam.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#include "OriginalConnNameParam.hpp"
#include "columns/HyPerCol.hpp"
#include "components/ConnectionData.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

OriginalConnNameParam::OriginalConnNameParam(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

OriginalConnNameParam::~OriginalConnNameParam() {}

int OriginalConnNameParam::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

void OriginalConnNameParam::setObjectType() { mObjectType = "OriginalConnNameParam"; }

int OriginalConnNameParam::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_originalConnName(ioFlag);
   return PV_SUCCESS;
}

void OriginalConnNameParam::ioParam_originalConnName(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamStringRequired(
         ioFlag, name, "originalConnName", &mOriginalConnName);
}

} // namespace PV
