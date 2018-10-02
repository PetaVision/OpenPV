/*
 * OriginalLayerNameParam.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#include "OriginalLayerNameParam.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObserverTableComponent.hpp"

namespace PV {

OriginalLayerNameParam::OriginalLayerNameParam(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

OriginalLayerNameParam::~OriginalLayerNameParam() {}

int OriginalLayerNameParam::initialize(char const *name, HyPerCol *hc) {
   return LinkedObjectParam::initialize(name, hc, std::string("originalLayerName"));
}

void OriginalLayerNameParam::setObjectType() { mObjectType = "OriginalLayerNameParam"; }

} // namespace PV
