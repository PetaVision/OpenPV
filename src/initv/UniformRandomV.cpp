/*
 * UniformRandomV.cpp
 *
 *  Created on: Oct 26, 2016
 *      Author: pschultz
 */

#include "UniformRandomV.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/Random.hpp"
#include "utils/PVLog.hpp"

namespace PV {

UniformRandomV::UniformRandomV() { initialize_base(); }

UniformRandomV::UniformRandomV(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

UniformRandomV::~UniformRandomV() {}

int UniformRandomV::initialize_base() { return PV_SUCCESS; }

int UniformRandomV::initialize(char const *name, HyPerCol *hc) {
   int status = BaseInitV::initialize(name, hc);
   return status;
}

int UniformRandomV::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseInitV::ioParamsFillGroup(ioFlag);
   ioParam_minV(ioFlag);
   ioParam_maxV(ioFlag);
   return status;
}

void UniformRandomV::ioParam_minV(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "minV", &minV, minV);
}

void UniformRandomV::ioParam_maxV(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "minV"));
   parent->parameters()->ioParamValue(ioFlag, name, "maxV", &maxV, minV + 1.0f);
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
