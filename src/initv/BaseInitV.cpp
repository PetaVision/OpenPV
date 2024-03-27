/*
 * BaseInitV.cpp
 *
 *  Created on: Oct 25, 2016
 *      Author: pschultz
 */

#include "BaseInitV.hpp"

namespace PV {

BaseInitV::BaseInitV() { initialize_base(); }

BaseInitV::BaseInitV(char const *name, PVParams *params, Communicator const *comm) {
   initialize_base();
   initialize(name, params, comm);
}

BaseInitV::~BaseInitV() {}

int BaseInitV::initialize_base() { return PV_SUCCESS; }

void BaseInitV::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseObject::initialize(name, params, comm);
}

void BaseInitV::setObjectType() {
   auto *params                = parameters();
   char const *initVTypeString = params->stringValue(getName(), "InitVType", false);
   mObjectType                 = initVTypeString ? std::string(initVTypeString) : mDefaultInitV;
}

int BaseInitV::ioParamsFillGroup(enum ParamsIOFlag ioFlag) { return PV_SUCCESS; }

void BaseInitV::calcV(float *V, PVLayerLoc const *loc) {}

std::string const BaseInitV::mDefaultInitV = "ConstantV";

} // end namespace PV
