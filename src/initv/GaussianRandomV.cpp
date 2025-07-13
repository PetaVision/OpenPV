/*
 * GaussianRandomV.cpp
 *
 *  Created on: Oct 26, 2016
 *      Author: pschultz
 */

#include "GaussianRandomV.hpp"
#include "columns/GaussianRandom.hpp"

namespace PV {

GaussianRandomV::GaussianRandomV() { initialize_base(); }

GaussianRandomV::GaussianRandomV(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize_base();
   initialize(params, defaults, comm);
}

GaussianRandomV::~GaussianRandomV() {}

int GaussianRandomV::initialize_base() { return PV_SUCCESS; }

void GaussianRandomV::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   BaseInitV::initialize(params, defaults, comm);
}

int GaussianRandomV::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = BaseInitV::ioParamsFillGroup(ioSwitch);
   ioParam_meanV(ioSwitch);
   ioParam_sigmaV(ioSwitch);
   return status;
}

void GaussianRandomV::ioParam_meanV(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "meanV", &meanV);
}

void GaussianRandomV::ioParam_sigmaV(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "maxV", &sigmaV);
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
