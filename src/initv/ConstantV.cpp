/*
 * ConstantV.cpp
 *
 *  Created on: Oct 26, 2011
 *      Author: pschultz
 */

#include "ConstantV.hpp"
#include "include/default_params.h"

namespace PV {

ConstantV::ConstantV() { initialize_base(); }

ConstantV::ConstantV(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize_base();
   initialize(params, defaults, comm);
}

ConstantV::~ConstantV() {}

int ConstantV::initialize_base() { return PV_SUCCESS; }

void ConstantV::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   BaseInitV::initialize(params, defaults, comm);
}

int ConstantV::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = BaseInitV::ioParamsFillGroup(ioSwitch);
   ioParam_valueV(ioSwitch);
   return status;
}

void ConstantV::ioParam_valueV(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "valueV", &mValueV);
}

void ConstantV::calcV(float *V, PVLayerLoc const *loc) {
   if (V == NULL) {
      Fatal().printf("%s: calcV called but membrane potential V is null.\n", getDescription_c());
   }
   int const numNeurons = loc->nx * loc->ny * loc->nf * loc->nbatch;
   for (int k = 0; k < numNeurons; k++) {
      V[k] = mValueV;
   }
}

} // end namespace PV
