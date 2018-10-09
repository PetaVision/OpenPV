/*
 * SingleArbor.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: pschultz
 */

#include "SingleArbor.hpp"
#include "columns/HyPerCol.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

SingleArbor::SingleArbor(char const *name, HyPerCol *hc) { initialize(name, hc); }

SingleArbor::SingleArbor() {}

SingleArbor::~SingleArbor() {}

int SingleArbor::initialize(char const *name, HyPerCol *hc) {
   return ArborList::initialize(name, hc);
}

void SingleArbor::setObjectType() { mObjectType = "SingleArbor"; }

int SingleArbor::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   return ArborList::ioParamsFillGroup(ioFlag);
}

void SingleArbor::ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      mNumAxonalArbors = 1;
      parameters()->handleUnnecessaryParameter(name, "numAxonalArbors", mNumAxonalArbors);
   }
}

} // namespace PV
