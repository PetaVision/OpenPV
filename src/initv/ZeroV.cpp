/*
 * ZeroV.cpp
 *
 *  Created on: Oct 26, 2011
 *      Author: pschultz
 */

#include "ZeroV.hpp"

namespace PV {
ZeroV::ZeroV() { initialize_base(); }

ZeroV::ZeroV(char const *name, PVParams *params, Communicator const *comm) {
   initialize_base();
   initialize(name, params, comm);
}

ZeroV::~ZeroV() {}

int ZeroV::initialize_base() { return PV_SUCCESS; }

void ZeroV::initialize(char const *name, PVParams *params, Communicator const *comm) {
   ConstantV::initialize(name, params, comm);
}

void ZeroV::ioParam_valueV(enum ParamsIOFlag ioFlag) {
   mValueV = 0.0f;
   if (ioFlag == PARAMS_IO_READ) {
      parameters()->handleUnnecessaryParameter(getName(), "valueV", 0.0f /*correctValue*/);
   }
}

} // end namespace PV
