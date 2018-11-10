/*
 * InputLayerNameParam.cpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#include "InputLayerNameParam.hpp"
#include "columns/HyPerCol.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

InputLayerNameParam::InputLayerNameParam(char const *name, HyPerCol *hc) { initialize(name, hc); }

InputLayerNameParam::~InputLayerNameParam() {}

int InputLayerNameParam::initialize(char const *name, HyPerCol *hc) {
   return LinkedObjectParam::initialize(name, hc, std::string("inputLayerName"));
}

void InputLayerNameParam::setObjectType() { mObjectType = "InputLayerNameParam"; }

} // namespace PV
