/*
 * ConstantV.cpp
 *
 *  Created on: Oct 26, 2011
 *      Author: pschultz
 */

#include "ConstantV.hpp"
#include "columns/HyPerCol.hpp"
#include "include/default_params.h"

namespace PV {

ConstantV::ConstantV() {
   initialize_base();
}

ConstantV::ConstantV(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

ConstantV::~ConstantV() {
}

int ConstantV::initialize_base() {
   return PV_SUCCESS;
}

int ConstantV::initialize(char const *name, HyPerCol *hc) {
   int status = BaseInitV::initialize(name, hc);
   return status;
}

int ConstantV::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseInitV::ioParamsFillGroup(ioFlag);
   ioParam_valueV(ioFlag);
   return status;
}

void ConstantV::ioParam_valueV(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "valueV", &mValueV, (pvdata_t)V_REST);
}

int ConstantV::calcV(pvdata_t *V, PVLayerLoc const *loc) {
   int status            = PV_SUCCESS;
   if (V == NULL) {
      pvError().printf(
            "%s: calcV called but membrane potential V is null.\n", getDescription_c());
   }
   int const numNeurons = loc->nx * loc->ny * loc->nf * loc->nbatch;
   for (int k = 0; k < numNeurons; k++) { V[k] = mValueV; }
   return PV_SUCCESS;
}

} // end namespace PV
