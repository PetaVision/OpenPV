/*
 * GaussianRandomV.cpp
 *
 *  Created on: Oct 26, 2016
 *      Author: pschultz
 */

#include "GaussianRandomV.hpp"
#include "columns/GaussianRandom.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

GaussianRandomV::GaussianRandomV() { initialize_base(); }

GaussianRandomV::GaussianRandomV(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

GaussianRandomV::~GaussianRandomV() {}

int GaussianRandomV::initialize_base() { return PV_SUCCESS; }

int GaussianRandomV::initialize(char const *name, HyPerCol *hc) {
   int status = BaseInitV::initialize(name, hc);
   return status;
}

int GaussianRandomV::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseInitV::ioParamsFillGroup(ioFlag);
   ioParam_meanV(ioFlag);
   ioParam_sigmaV(ioFlag);
   return status;
}

void GaussianRandomV::ioParam_meanV(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "meanV", &meanV, meanV);
}

void GaussianRandomV::ioParam_sigmaV(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "maxV", &sigmaV, sigmaV);
}

void GaussianRandomV::calcV(float *V, PVLayerLoc const *loc) {
   PVLayerLoc flatLoc;
   memcpy(&flatLoc, loc, sizeof(PVLayerLoc));
   flatLoc.nf = 1;
   GaussianRandom randState{&flatLoc, false /*not extended*/};
   const int nxny = flatLoc.nx * flatLoc.ny;
   for (int b = 0; b < loc->nbatch; b++) {
      float *VBatch = V + b * loc->nx * loc->ny * loc->nf;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int xy = 0; xy < nxny; xy++) {
         for (int f = 0; f < loc->nf; f++) {
            int index     = kIndex(xy, 0, f, nxny, 1, loc->nf);
            VBatch[index] = randState.gaussianDist(xy, meanV, sigmaV);
         }
      }
   }
}

} // end namespace PV
