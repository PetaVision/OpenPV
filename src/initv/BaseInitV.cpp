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

BaseInitV::BaseInitV(char const *name, PVParams *params, Communicator *comm) {
   initialize_base();
   initialize(name, params, comm);
}

BaseInitV::~BaseInitV() {}

int BaseInitV::initialize_base() { return PV_SUCCESS; }

void BaseInitV::initialize(char const *name, PVParams *params, Communicator *comm) {
   BaseObject::initialize(name, params, comm);
}

void BaseInitV::setObjectType() {
   auto *params                = parameters();
   char const *initVTypeString = params->stringValue(name, "InitVType", false);
   mObjectType                 = initVTypeString ? std::string(initVTypeString) : mDefaultInitV;
}

int BaseInitV::ioParamsFillGroup(enum ParamsIOFlag ioFlag) { return PV_SUCCESS; }

void BaseInitV::calcV(float *V, PVLayerLoc const *loc) {}

string const BaseInitV::mDefaultInitV = "ConstantV";

} // end namespace PV
