/*
 * ZeroV.cpp
 *
 *  Created on: Oct 26, 2011
 *      Author: pschultz
 */

#include "ZeroV.hpp"

namespace PV {
ZeroV::ZeroV() { initialize_base(); }

ZeroV::ZeroV(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize_base();
   initialize(paramsIO, comm);
}

ZeroV::~ZeroV() {}

int ZeroV::initialize_base() { return PV_SUCCESS; }

void ZeroV::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   ConstantV::initialize(paramsIO, comm);
}

void ZeroV::ioParam_valueV(ParamsIOSwitch ioSwitch) {
   mValueV = 0.0f;
   if (ioSwitch == ParamsIOSwitch::Read) {
      mParamsIO->handleUnnecessaryParameter("valueV", 0.0f /*correctValue*/);
   }
}

} // end namespace PV
