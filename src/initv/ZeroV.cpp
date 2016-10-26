/*
 * ZeroV.cpp
 *
 *  Created on: Oct 26, 2011
 *      Author: pschultz
 */

#include "ZeroV.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {
ZeroV::ZeroV() {
   initialize_base();
}

ZeroV::ZeroV(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

ZeroV::~ZeroV() {
}

int ZeroV::initialize_base() {
   return PV_SUCCESS;
}

int ZeroV::initialize(char const *name, HyPerCol *hc) {
   int status = ConstantV::initialize(name, hc);
   return status;
}

void ZeroV::ioParam_valueV(enum ParamsIOFlag ioFlag) {
   mValueV = 0.0f;
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "valueV", 0.0f /*correctValue*/);
   }
}

} // end namespace PV
