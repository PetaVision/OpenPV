/*
 * UniformRandomV.cpp
 *
 *  Created on: Oct 26, 2016
 *      Author: pschultz
 */

#include "UniformRandomV.hpp"
#include "columns/Random.hpp"
#include "utils/PVLog.hpp"

namespace PV {

UniformRandomV::UniformRandomV() { initialize_base(); }

UniformRandomV::UniformRandomV(char const *name, PVParams *params, Communicator const *comm) {
   initialize_base();
   initialize(name, params, comm);
}

UniformRandomV::~UniformRandomV() {}

int UniformRandomV::initialize_base() { return PV_SUCCESS; }

void UniformRandomV::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseInitV::initialize(name, params, comm);
}

int UniformRandomV::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseInitV::ioParamsFillGroup(ioFlag);
   ioParam_minV(ioFlag);
   ioParam_maxV(ioFlag);
   return status;
}

void UniformRandomV::ioParam_minV(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "minV", &minV, minV);
}

void UniformRandomV::ioParam_maxV(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "minV"));
   parameters()->ioParamValue(ioFlag, name, "maxV", &maxV, minV + 1.0f);
}

void UniformRandomV::calcV(float *V, PVLayerLoc const *loc) {
   PVLayerLoc flatLoc;
   memcpy(&flatLoc, loc, sizeof(PVLayerLoc));
   flatLoc.nf = 1;
   Random randState{&flatLoc, false /*not extended*/};
   const int nxny = flatLoc.nx * flatLoc.ny;
   for (int b = 0; b < loc->nbatch; b++) {
      float *VBatch = V + b * loc->nx * loc->ny * loc->nf;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int xy = 0; xy < nxny; xy++) {
         for (int f = 0; f < loc->nf; f++) {
            int index     = kIndex(xy, 0, f, nxny, 1, loc->nf);
            VBatch[index] = randState.uniformRandom(xy, minV, maxV);
         }
      }
   }
}

} // end namespace PV
