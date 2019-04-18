/*
 * BaseInitV.cpp
 *
 *  Created on: Oct 25, 2016
 *      Author: pschultz
 */

#include "BaseInitV.hpp"
#include "columns/HyPerCol.hpp" // To get params for setObjectType

namespace PV {

BaseInitV::BaseInitV() { initialize_base(); }

BaseInitV::BaseInitV(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

BaseInitV::~BaseInitV() {}

int BaseInitV::initialize_base() { return PV_SUCCESS; }

int BaseInitV::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

void BaseInitV::setObjectType() {
   auto *params                = parent->parameters();
   char const *initVTypeString = params->stringValue(name, "InitVType", false);
   mObjectType                 = initVTypeString ? std::string(initVTypeString) : mDefaultInitV;
}

int BaseInitV::ioParamsFillGroup(enum ParamsIOFlag ioFlag) { return PV_SUCCESS; }

void BaseInitV::calcV(float *V, PVLayerLoc const *loc) {}

string const BaseInitV::mDefaultInitV = "ConstantV";

} // end namespace PV
