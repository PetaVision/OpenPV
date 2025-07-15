/*
 * BaseInitV.cpp
 *
 *  Created on: Oct 25, 2016
 *      Author: pschultz
 */

#include "BaseInitV.hpp"
#include "params/ParamsIO.hpp"

namespace PV {

BaseInitV::BaseInitV() { initialize_base(); }

BaseInitV::BaseInitV(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize_base();
   initialize(paramsIO, comm);
}

BaseInitV::~BaseInitV() {}

int BaseInitV::initialize_base() { return PV_SUCCESS; }

void BaseInitV::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   BaseObject::initialize(paramsIO, comm);
}

void BaseInitV::setObjectType() {
   std::string mObjectType = mParamsIO->readValue<std::string>("InitVType");
   FatalIf(
         mObjectType.empty(),
         "InitVType for parameter group \"%s\" cannot be NULL or empty.\n", getName());
}

int BaseInitV::ioParamsFillGroup(ParamsIOSwitch ioSwitch) { return PV_SUCCESS; }

void BaseInitV::calcV(float *V, PVLayerLoc const *loc) {}

} // end namespace PV
