/*
 * OriginalConnNameParam.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#include "OriginalConnNameParam.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObserverTableComponent.hpp"

namespace PV {

OriginalConnNameParam::OriginalConnNameParam(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

OriginalConnNameParam::~OriginalConnNameParam() {}

int OriginalConnNameParam::initialize(char const *name, HyPerCol *hc) {
   return LinkedObjectParam::initialize(name, hc, std::string("originalConnName"));
}

void OriginalConnNameParam::setObjectType() { mObjectType = "OriginalConnNameParam"; }

} // namespace PV
