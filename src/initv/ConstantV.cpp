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

ConstantV::ConstantV(char const *name, PVParams *params, Communicator const *comm) {
   initialize_base();
   initialize(name, params, comm);
}

ConstantV::~ConstantV() {}

int ConstantV::initialize_base() { return PV_SUCCESS; }

void ConstantV::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseInitV::initialize(name, params, comm);
}

int ConstantV::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseInitV::ioParamsFillGroup(ioFlag);
   ioParam_valueV(ioFlag);
   return status;
}

void ConstantV::ioParam_valueV(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "valueV", &mValueV, (float)V_REST);
}

void ConstantV::calcV(float *V, PVLayerLoc const *loc) {
   if (V == NULL) {
      Fatal().printf("%s: calcV called but membrane potential V is null.\n", getDescription_c());
   }
   long const numNeurons = (long)loc->nx * (long)loc->ny * (long)loc->nf * (long)loc->nbatch;
   for (long k = 0; k < numNeurons; k++) {
      V[k] = mValueV;
   }
}

} // end namespace PV
