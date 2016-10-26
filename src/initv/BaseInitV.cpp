/*
 * BaseInitV.cpp
 *
 *  Created on: Oct 25, 2016
 *      Author: pschultz
 */

#include "BaseInitV.hpp"

namespace PV {

BaseInitV::BaseInitV() {
   initialize_base();
}

BaseInitV::BaseInitV(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

BaseInitV::~BaseInitV() {
}

int BaseInitV::initialize_base() {
   return PV_SUCCESS;
}

int BaseInitV::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

int BaseInitV::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   return PV_SUCCESS;
}

int BaseInitV::calcV(pvdata_t *V, PVLayerLoc const *loc) {
   return PV_SUCCESS;
}

} // end namespace PV
